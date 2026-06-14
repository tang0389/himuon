import math

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from himuon.triton_kernels import XXT, ba_plus_cAA, fused_bmm_add


class HiMuonLegacy(Optimizer):
    """
    Hierarchical Muon with tile-local Newton-Schulz.

    Params with ``numel <= tile_h * tile_w`` fall back to full-matrix NS
    (shape-driven, not time-driven). To switch tile_size or other group
    defaults mid-training, call ``optimizer.reconfigure(...)``.

    Args:
        params: model parameters
        lr: learning rate (default 0.02)
        momentum: momentum coefficient (default 0.95)
        nesterov: use Nesterov momentum (default True)
        weight_decay: weight decay (default 0.1)
        tile_size: tile size for tiled NS (default 512).
            int → square tiles (T, T); tuple (Th, Tw) → non-square tiles.
        ns_steps: Newton-Schulz iterations (default 5)
        use_muon: use Muon path, otherwise AdamW fallback (default True)
        lr_adjust: Muon-style LR scaling by block shape, "tile" or "none" (default "tile")
        adamw_betas: betas for AdamW fallback (default (0.95, 0.95))
        adamw_eps: epsilon for AdamW fallback (default 1e-8)
    """

    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        weight_decay=0.1,
        tile_size=512,
        ns_steps=5,
        use_muon=True,
        lr_adjust="tile",
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
    ):
        tile_size = self._normalize_defaults(tile_size)

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            tile_size=tile_size,
            ns_steps=ns_steps,
            use_muon=use_muon,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        super().__init__(params, defaults)

        if lr_adjust not in ("none", "tile"):
            raise ValueError(f"lr_adjust must be 'none' or 'tile', got {lr_adjust!r}")
        self.lr_adjust = lr_adjust

    # -- Reconfiguration --------------------------------------------------

    @staticmethod
    def _normalize_defaults(tile_size):
        if isinstance(tile_size, int):
            tile_size = (tile_size, tile_size)
        tile_size = tuple(tile_size)
        assert len(tile_size) == 2 and all(t > 0 for t in tile_size)
        return tile_size

    def reconfigure(self, **updates):
        """Update param-group defaults mid-training (e.g., tile_size).
        Invalidates any cached bucket plan or CUDA graph via the
        ``_invalidate_plan_cache`` hook (subclasses override).

        Example:
            opt = HiMuon(params, tile_size=1_000_000)  # full-matrix NS
            for _ in range(100): train_step(opt)
            opt.reconfigure(tile_size=512)             # switch to tiled
            for _ in range(N): train_step(opt)
        """
        for group in self.param_groups:
            ts = self._normalize_defaults(updates.get("tile_size", group["tile_size"]))
            for k, v in updates.items():
                if k in group:
                    group[k] = v
            if "tile_size" in updates:
                group["tile_size"] = ts
        self._invalidate_plan_cache()

    def _invalidate_plan_cache(self):
        """Hook for subclasses to flush cached plan / CUDA graph on
        ``reconfigure``. No-op in the baseline."""
        pass

    # -- Newton-Schulz (Triton-accelerated) --------------------------------

    @staticmethod
    @torch.compile(dynamic=False, fullgraph=True)
    def newton_schulz(G, steps=5):
        assert G.ndim >= 2
        a, b, c = (3.4445, -4.7750, 2.0315)
        original_dtype = G.dtype
        X = G.bfloat16()
        if G.size(-2) > G.size(-1):
            X = X.transpose(-2, -1)
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        X = X.contiguous()
        A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
        B = torch.empty_like(A)
        C = torch.empty_like(X)
        for _ in range(steps):
            XXT(X, out=A)
            ba_plus_cAA(A, alpha=c, beta=b, out=B)
            if X.ndim > 2:
                fused_bmm_add(B, X, a, out=C)
            else:
                torch.addmm(X, B, X, beta=a, alpha=1.0, out=C)
            X, C = C, X
        if G.size(-2) > G.size(-1):
            X = X.transpose(-2, -1)
        return X.to(original_dtype)

    # -- Tiling ------------------------------------------------------------

    @staticmethod
    def _tile(tensor, tile_size):
        """Tile (H, W) → (R, C, Th, Tw) with zero-padding."""
        tile_h, tile_w = tile_size
        h, w = tensor.shape
        pad_h = (tile_h - h % tile_h) % tile_h
        pad_w = (tile_w - w % tile_w) % tile_w
        if pad_h or pad_w:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h))
        H, W = tensor.shape
        R, C = H // tile_h, W // tile_w
        tiled = tensor.view(R, tile_h, C, tile_w).permute(0, 2, 1, 3).contiguous()
        return tiled, (h, w, pad_h, pad_w)

    @staticmethod
    def _untile(tiled, info):
        """Restore (R, C, Th, Tw) → (H, W) and remove padding."""
        h, w, pad_h, pad_w = info
        R, C, Th, Tw = tiled.shape
        restored = tiled.permute(0, 2, 1, 3).contiguous().view(R * Th, C * Tw)
        if pad_h or pad_w:
            restored = restored[:h, :w]
        return restored

    # -- Learning-rate adjustment ------------------------------------------

    def adjust_lr_for_muon(self, lr, block_shape):
        """Adapted from Muon's adjust_lr_for_muon."""
        if self.lr_adjust == "none":
            return lr
        a, b = block_shape[:2]
        return lr * 0.2 * math.sqrt(max(a, b))

    # -- Helpers -----------------------------------------------------------

    def _compute_momentum(self, p, g, momentum, nesterov):
        if "momentum_buffer" not in self.state[p]:
            self.state[p]["momentum_buffer"] = torch.zeros_like(g)
        buf = self.state[p]["momentum_buffer"]
        buf.mul_(momentum).add_(g)
        if nesterov:
            return g.add(buf, alpha=momentum)
        return buf

    def _compute_adamw_step(self, p, g, betas, eps):
        if "step" not in self.state[p]:
            self.state[p]["step"] = 0
            self.state[p]["moment1"] = torch.zeros_like(g)
            self.state[p]["moment2"] = torch.zeros_like(g)
        self.state[p]["step"] += 1
        step = self.state[p]["step"]
        buf1, buf2 = self.state[p]["moment1"], self.state[p]["moment2"]
        buf1.lerp_(g, 1 - betas[0])
        buf2.lerp_(g.square(), 1 - betas[1])
        bc1 = 1 - betas[0] ** step
        bc2 = 1 - betas[1] ** step
        return (buf1 / bc1) / (buf2.sqrt() / bc2**0.5 + eps)

    # -- Step --------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            tile_h, tile_w = group["tile_size"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                if group["use_muon"]:
                    g = self._compute_momentum(
                        p, grad, group["momentum"], group["nesterov"]
                    )

                    if grad.numel() <= tile_h * tile_w:
                        update = self.newton_schulz(g, steps=ns_steps)
                        block = p.shape
                    else:
                        tiled, info = self._tile(g, (tile_h, tile_w))
                        R, C, Th, Tw = tiled.shape
                        orth = self.newton_schulz(
                            tiled.view(-1, Th, Tw), steps=ns_steps
                        ).view(R, C, Th, Tw)
                        update = self._untile(orth, info)
                        block = (tile_h, tile_w)

                    step_lr = self.adjust_lr_for_muon(group["lr"], block)
                else:
                    update = self._compute_adamw_step(
                        p, grad, group["adamw_betas"], group["adamw_eps"]
                    )
                    step_lr = group["lr"]

                if group["weight_decay"] > 0.0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])
                p.data.add_(update, alpha=-step_lr)

        return loss
