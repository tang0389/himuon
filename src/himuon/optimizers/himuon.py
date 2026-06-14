import math
import warnings

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from himuon.triton_kernels import (
    XXT,
    ba_plus_cAA,
    fused_bmm_add,
    ns5_smem,
)

# Optional DTensor guard (safe if distributed tensor is unavailable).
try:
    from torch.distributed.tensor import DTensor as _DTensor  # type: ignore
except Exception:  # pragma: no cover - torch version without DTensor
    _DTensor = None  # type: ignore


class HiMuon(Optimizer):
    """
    Hierarchical Muon with cross-layer batched NS.
    Runs on single GPU, DDP, and FSDP2.

    Semantics:
      * Per-param pre-NS work (momentum, tile) runs sequentially.
      * Tiles from all params in a bucket are copied into a single concat
        buffer keyed by ``(tile_h, tile_w, full_ns_flag=False, dtype)``.
      * A **single batched NS call per chunk of ≤ B_hw tiles** processes
        all tiles at once. Per-tile math is algebraically independent.
      * Post-NS work (untile, WD, ``p.data.add_``) runs per-param after
        scatter-back from the concat buffer.

    Plan / buffer caching:
      * ``_plan_buckets()`` output is cached in ``self._cached_plan``
        across steps and only rebuilt on ``reconfigure()``.
      * Concat buffers in ``self._concat_buffers`` are allocated once and
        grown monotonically — pointers stay stable across steps, which
        CUDA graph capture relies on.

    FSDP2 support:
      DTensor parameters (from ``fully_shard``) are handled natively by
      operating on the local row-shard directly. Per-tile independence
      means each rank runs NS on its own tiles with no all-gather /
      all-reduce — tile output depends only on the tile itself.

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
        adamw_betas: betas for AdamW fallback (default (0.95, 0.95))
        adamw_eps: epsilon for AdamW fallback (default 1e-8)
        b_hw: cross-layer batch size (int). ``None`` → auto-derived
            from the GPU memory budget (see ``_auto_b_hw``).
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
        b_hw=None,
        cuda_graph=True,
        cuda_graph_warmup=3,
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

        # Per-instance cross-layer batching state.
        # ``b_hw`` is a per-call chunk-size cap (user override). Auto mode
        # derives the cap from available GPU memory so each NS call hits
        # the largest batch the hardware can hold — this is what lets
        # Gigathread saturate SMs across the whole bucket with a single
        # kernel launch (or as few as memory allows).
        self._b_hw_override = b_hw
        # Memory budget for NS intermediates per call. NS internally
        # allocates X + A + B + C (4 tensors of shape (B, T, T) each),
        # so budget = 4 × B_max × T² × dtype_bytes. Default: 15% of
        # total GPU memory.
        self._memory_budget_fraction = 0.15
        self._concat_buffers: dict[tuple, torch.Tensor] = {}
        self._cached_plan: "list[dict] | None" = None
        self._last_step_stats: dict = {
            "ns_gpu_ms": 0.0,
            "ns_launches": 0,
            "total_B": 0,
            "avg_batch": 0.0,
        }

        # CUDA graph capture of the full step. After
        # ``cuda_graph_warmup`` eager steps (to let Triton autotune and
        # caching-allocator settle), capture the step body into a graph
        # and replay it on subsequent step() calls. Collapses
        # small per-param kernel launches per step into one
        # ``cudaGraphLaunch`` command.
        #
        # Caveats:
        #   * Users MUST call ``zero_grad(set_to_none=False)`` so grad
        #     tensors keep stable memory addresses across steps. The
        #     captured graph references grad pointers directly.
        #   * LR / weight_decay are applied OUTSIDE the graph (see the
        #     narrow-scope note below), so LR schedulers take effect
        #     on the next step without recapture.
        #   * In-graph region disables CUDA-event NS timing
        #     (``synchronize()`` is illegal inside capture), so
        #     ``_last_step_stats["ns_gpu_ms"]`` reports 0 for replay
        #     steps. Measure via outer event timing.
        self._cuda_graph_enabled = bool(cuda_graph)
        self._cuda_graph_warmup = int(cuda_graph_warmup)
        self._cuda_graph: "torch.cuda.CUDAGraph | None" = None
        self._graph_step_count: int = 0
        self._in_graph_region: bool = False

        if self._cuda_graph_enabled and any(
            self._is_dtensor(p) for g in self.param_groups for p in g["params"]
        ):
            warnings.warn(
                "HiMuon under FSDP2 with cuda_graph=True: graph pool and "
                "FSDP all-gather buffers contend for memory, slowing "
                "fwd/bwd. Pass cuda_graph=False, or set "
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True in the "
                "shell before launch.",
                stacklevel=2,
            )

        # Narrow graph scope: the captured region computes the per-param
        # ``update`` tensor (through momentum → tile → NS → scatter →
        # untile) but does NOT apply ``p.data -= lr*update``. The
        # WD+LR application runs in Python with scalar ``group["lr"]``
        # and ``group["weight_decay"]`` AFTER ``graph.replay()``.
        # This keeps LR schedulers natively compatible — changing
        # ``group["lr"]`` takes effect on the very next step, with no
        # recapture and no tensor-pointer plumbing.
        #
        # Populated during capture (Python-side list of
        # (param, update_tensor, group_idx)); tensor refs point into
        # graph-pool memory whose contents refresh on each replay.
        self._pending_updates: list[tuple] = []

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

    def adjust_lr_for_muon(self, lr, block_shape):
        """Adapted from Muon's adjust_lr_for_muon."""
        if self.lr_adjust == "none":
            return lr
        a, b = block_shape[:2]
        return lr * 0.2 * math.sqrt(max(a, b))

    def reconfigure(self, **updates):
        """Update param-group defaults mid-training (e.g., tile_size).
        Invalidates any cached bucket plan + concat buffer via the
        ``_invalidate_plan_cache`` hook so the next step rebuilds fresh
        state.
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
        """Flush cached plan, concat buffers, CUDA graph, and the pending
        updates list (which references graph-pool tensors that become
        invalid once the graph is dropped).
        """
        self._concat_buffers.clear()
        self._cached_plan = None
        self._cuda_graph = None
        self._graph_step_count = 0
        self._pending_updates.clear()
        torch._dynamo.reset()

    def _apply_pending_updates(self):
        """Apply WD + LR-scaled update to params. Called AFTER
        ``graph.replay()`` populates the ``update`` tensors. Reads
        current scalar ``group["lr"]`` / ``group["weight_decay"]``
        each call, so LR-scheduler mutations take effect immediately
        without recapture.
        """
        for p, update, group_idx in self._pending_updates:
            group = self.param_groups[group_idx]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            if weight_decay > 0.0:
                p.data.mul_(1 - lr * weight_decay)
            self._write_update_inplace(p, update, alpha=-lr)

    # -- Newton-Schulz (dispatched on tile size) --------------------------

    @staticmethod
    @torch.compile(dynamic=False, fullgraph=True)
    def _newton_schulz_3kernel(G, steps=5):
        """3-kernel NS path (T >= 256 or 2D full-NS). Uses XXT + ba_plus_cAA
        + fused_bmm_add Triton kernels, @torch.compile'd for fusion."""
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

    # Dims where the fused NS5 kernel beats the 3-kernel path. Admitted
    # shapes are pairs drawn from this set with product ≤ 16384.
    _NS5_DIMS = frozenset({16, 32, 64, 128, 256, 512, 1024})

    @staticmethod
    def newton_schulz(G, steps=5):
        assert G.ndim >= 2

        if (
            G.ndim == 3
            and G.size(-2) in HiMuon._NS5_DIMS
            and G.size(-1) in HiMuon._NS5_DIMS
            and G.size(-2) * G.size(-1) <= 16384
        ):
            original_dtype = G.dtype
            X = G.bfloat16()
            transposed = False
            if G.size(-2) > G.size(-1):
                X = X.transpose(-2, -1)
                transposed = True
            X = X.contiguous()
            X = ns5_smem(X)
            if transposed:
                X = X.transpose(-2, -1)
            return X.to(original_dtype)
        return HiMuon._newton_schulz_3kernel(G, steps=steps)

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

    # WD/LR application is intentionally OUTSIDE this helper so the LR
    # scheduler can mutate ``group["lr"]`` between steps without forcing
    # a recompile.
    @staticmethod
    @torch.compile(dynamic=True, fullgraph=True)
    def _post_ns_per_param(
        orth_flat,
        R: int,
        C: int,
        Th: int,
        Tw: int,
        h: int,
        w: int,
        target_dtype,
    ):
        orth = orth_flat.to(target_dtype).view(R, C, Th, Tw)
        restored = orth.permute(0, 2, 1, 3).contiguous().view(R * Th, C * Tw)
        return restored[:h, :w]

    # -- Helpers -----------------------------------------------------------

    def _apply_wd_and_update(self, p, update, weight_decay, lr, group_idx, block=None):
        """Apply WD + LR update. ``block`` = NS input shape for muon (scaled by
        ``adjust_lr_for_muon``); ``None`` (AdamW) keeps raw lr.
        """
        if self._in_graph_region:
            scale = self.adjust_lr_for_muon(1.0, block) if block is not None else 1.0
            self._pending_updates.append(
                (p, update if scale == 1.0 else update.mul_(scale), group_idx)
            )
        else:
            if weight_decay > 0.0:
                p.data.mul_(1 - lr * weight_decay)
            adjusted_lr = (
                self.adjust_lr_for_muon(lr, block) if block is not None else lr
            )
            self._write_update_inplace(p, update, alpha=-adjusted_lr)

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

    # -- Chunk-size cap (memory-bounded) ----------------------------------

    def _auto_b_hw(self, T: int, dtype_bytes: int = 4) -> int:
        """Max chunk size per NS call for a given tile edge T.

        The cap is the minimum of:
          * ``self._b_hw_override`` (if user passed it)
          * memory-derived max: ``budget // (4 × T² × dtype_bytes)`` where
            budget is ``_memory_budget_fraction × total_gpu_memory``.
            The "4 ×" covers the NS intermediates X + A + B + C.

        The memory-derived cap is the *real* constraint: GPU Gigathread
        handles arbitrarily large batches inside a single kernel launch,
        so smaller-than-memory caps would only add per-kernel overhead
        without any GPU-side benefit.
        """
        if not torch.cuda.is_available():
            # CPU fallback: arbitrary small cap to keep the bench happy.
            user = self._b_hw_override
            return max(1, int(user)) if user is not None else 1

        device = torch.cuda.current_device()
        total_mem = torch.cuda.get_device_properties(device).total_memory
        budget = int(self._memory_budget_fraction * total_mem)
        bytes_per_tile = 4 * T * T * dtype_bytes
        mem_cap = max(1, budget // bytes_per_tile)

        if self._b_hw_override is not None:
            return max(1, min(int(self._b_hw_override), mem_cap))
        return mem_cap

    # -- Public introspection ---------------------------------------------

    def xlayer_plan(self) -> list[dict]:
        """Return the cached bucket plan. Fresh optimizers (before the
        first step) return an empty list. Callers should treat the
        returned list as a snapshot — mutating it does not affect the
        optimizer.
        """
        if self._cached_plan is None:
            return []
        return list(self._cached_plan)

    # -- Plan / Execute ----------------------------------------------------

    def _is_dtensor(self, p) -> bool:
        return _DTensor is not None and isinstance(p, _DTensor)

    def _local_grad(self, p) -> torch.Tensor:
        """Return the local grad as a 2D tensor.

        3D DTensor bank (local shard ``(N_local, H, W)``): return a
        no-copy ``(N_local * H, W)`` reshape that stacks the N_local
        matrices along dim 0, so downstream 2D tile/NS code runs
        unchanged. This stacking preserves per-matrix tile semantics
        only when ``tile_h`` divides ``H`` — otherwise a tile would
        straddle two stacked matrices. The plan step validates this
        before any call here.
        """
        if self._is_dtensor(p):
            g = p.grad.to_local()
            if g.ndim == 3:
                N, H, W = g.shape
                return g.reshape(N * H, W)
            return g
        return p.grad

    def _write_update_inplace(self, p, update, alpha: float) -> None:
        """Apply ``p.data += alpha * update`` in place.

        3D DTensor bank: ``update`` arrives as the 2D ``(N_local*H, W)``
        shape used by the NS pipeline, while the local shard's
        underlying storage is ``(N_local, H, W)``. Reshape ``update``
        back to 3D before the add so elements line up with storage.
        """
        if self._is_dtensor(p):
            target = p.to_local()
            if target.ndim == 3:
                N, H, W = target.shape
                target.view(N * H, W).add_(update, alpha=alpha)
            else:
                target.add_(update, alpha=alpha)
        else:
            p.data.add_(update, alpha=alpha)

    @staticmethod
    def _balanced_chunks(B_total: int, b_hw: int) -> list[int]:
        """Split ``B_total`` into balanced chunks of size ≤ b_hw.

        Uses ``ceil(B_total / b_hw)`` chunks, each of size either
        ``ceil(remaining / remaining_chunks)`` so no chunk is much
        smaller than the rest. Unlike the remainder-style split
        (``while remaining > 0: c = min(b_hw, remaining)``), the last
        chunk here is at most 1 less than the first, not a tiny tail.

        Invariant: ``sum(chunks) == B_total`` and all chunks > 0.
        """
        if B_total <= 0:
            return []
        n_chunks = math.ceil(B_total / b_hw)
        chunks: list[int] = []
        remaining = B_total
        for i in range(n_chunks):
            remaining_chunks = n_chunks - i
            c = math.ceil(remaining / remaining_chunks)
            chunks.append(c)
            remaining -= c
        return chunks

    def _plan_buckets(self) -> list[dict]:
        """Walk all muon param groups, classify each param into a bucket by
        ``(tile_h, tile_w, full_ns_flag, dtype)``, and return a list of
        bucket plans.

        Only ``full_ns_flag == False`` buckets are returned: full-NS
        (single-tile) params run per-param through the dispatcher.
        """
        # Collect candidates by bucket_key. Each entry is a tuple
        # (param, group, tile_count).
        by_key: dict[tuple, list] = {}

        for group in self.param_groups:
            if not group["use_muon"]:
                continue
            tile_h, tile_w = group["tile_size"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                eff_tile_h, eff_tile_w = tile_h, tile_w

                is_dt = self._is_dtensor(p)
                if is_dt:
                    # Bank DTensor: sharded on layer axis N, so each
                    # rank holds complete (H, W) matrices locally.
                    local_raw = p.grad.to_local()
                    local_shape = tuple(local_raw.shape)

                    if local_raw.ndim != 3:
                        raise NotImplementedError(
                            f"HiMuon under FSDP2 requires bank sharding "
                            f"(3D DTensor on the layer axis). Got local "
                            f"ndim={local_raw.ndim}. Wrap the model with "
                            f"wrap_with_banks() before fully_shard()."
                        )
                    N_local, H_per, W = local_raw.shape
                    if tile_h >= H_per and tile_w >= W:
                        eff_tile_h, eff_tile_w = H_per, W
                    else:
                        if H_per % tile_h != 0:
                            raise RuntimeError(
                                f"HiMuon under FSDP2: bank DTensor "
                                f"requires tile_h to divide per-matrix H "
                                f"(tiles must not straddle matrix "
                                f"boundaries); got tile_h={tile_h}, "
                                f"H={H_per} (local shard {local_shape})."
                            )
                        if W % tile_w != 0:
                            raise RuntimeError(
                                f"HiMuon under FSDP2: bank DTensor "
                                f"requires tile_w to divide W; got "
                                f"tile_w={tile_w}, W={W}."
                            )
                    grad = local_raw.reshape(N_local * H_per, W)
                    h_local, w_local = grad.shape
                else:
                    grad = p.grad
                    full_ns = grad.numel() <= tile_h * tile_w
                    if full_ns:
                        # Full-NS path stays per-param (non-DTensor).
                        continue

                h, w = grad.shape
                pad_h = (eff_tile_h - h % eff_tile_h) % eff_tile_h
                pad_w = (eff_tile_w - w % eff_tile_w) % eff_tile_w
                R = (h + pad_h) // eff_tile_h
                C = (w + pad_w) // eff_tile_w
                tile_count = R * C

                # bucket_key: (tile_h, tile_w, full_ns_flag=False, dtype)
                key = (eff_tile_h, eff_tile_w, False, grad.dtype)
                by_key.setdefault(key, []).append((p, group, tile_count))

        plan: list[dict] = []
        for key, entries in by_key.items():
            tile_h, tile_w, _full_ns, _dtype = key
            T = max(tile_h, tile_w)
            dtype_bytes = torch.empty((), dtype=_dtype).element_size()
            b_hw = self._auto_b_hw(T, dtype_bytes=dtype_bytes)

            param_ids = [id(e[0]) for e in entries]
            tile_count_per_param = [e[2] for e in entries]
            B_total = sum(tile_count_per_param)

            chunk_sizes = self._balanced_chunks(B_total, b_hw)

            plan.append(
                {
                    "bucket_key": key,
                    "param_ids": param_ids,
                    "tile_count_per_param": tile_count_per_param,
                    "B_total": B_total,
                    "B_hw": b_hw,
                    "chunk_sizes": chunk_sizes,
                    "n_calls": len(chunk_sizes),
                    # Private — kept in the plan dict so _execute_plan can
                    # walk params without re-resolving id(p). Not part of
                    # the public introspection contract.
                    "_entries": entries,
                }
            )

        return plan

    def _ensure_concat_buffer(
        self, key: tuple, B: int, Th: int, Tw: int, device, dtype
    ) -> torch.Tensor:
        """Return a ``(B_at_least, Th, Tw)`` buffer for this bucket.

        Storage dtype follows the bucket's grad dtype (the last element
        of ``bucket_key``).

        Memory cost is bounded by the largest bucket size × tile area
        × dtype size; the buffer is reused across steps (monotonic
        growth, never freed).
        """
        store_dtype = dtype
        buf = self._concat_buffers.get(key)
        if (
            buf is None
            or buf.size(0) < B
            or buf.size(1) != Th
            or buf.size(2) != Tw
            or buf.device != device
            or buf.dtype != store_dtype
        ):
            buf = torch.empty((B, Th, Tw), device=device, dtype=store_dtype)
            self._concat_buffers[key] = buf
        return buf

    def _execute_plan(self, plan: list[dict], group_filter=None) -> None:
        """Execute the cross-layer batched NS plan.

        Pipeline per bucket (``full_ns_flag=False`` only; full-NS
        single-tile non-DTensor params handled in
        ``_step_muon_full_ns``):

          1. Per-param pre-NS pass: momentum → tile. Tiles copied into
             the bucket's concat buffer (storage dtype = grad.dtype; see
             ``_ensure_concat_buffer``).
          2. Batched NS over the concat buffer, walking
             ``chunk_sizes``. Each chunk gets one dispatcher call with
             shape ``(chunk, Th, Tw)``; result written back in place.
          3. Per-param post-NS pass: slice the concat buffer by the
             param's offset, reshape to ``(R, C, Th, Tw)``, divide by
             scale, untile, WD, ``p.data.add_``.
        """
        ns_gpu_ms = 0.0
        ns_launches = 0
        total_B = 0
        # Disable CUDA-event timing inside graph capture: the per-bucket
        # ``ev.synchronize()`` is illegal in a capture region, and
        # ``elapsed_time()`` requires sync. Eager mode keeps the events.
        use_events = torch.cuda.is_available() and not self._in_graph_region

        for bucket in plan:
            key = bucket["bucket_key"]
            tile_h, tile_w, _full_ns, _dtype = key
            entries = bucket["_entries"]
            tile_count_per_param = bucket["tile_count_per_param"]
            chunk_sizes = bucket["chunk_sizes"]
            B_total = bucket["B_total"]
            ns_steps_list = [e[1]["ns_steps"] for e in entries]
            assert len(set(ns_steps_list)) <= 1, (
                "cross-layer NS requires a uniform ns_steps per bucket; "
                f"saw {set(ns_steps_list)}"
            )
            ns_steps = ns_steps_list[0] if ns_steps_list else 5

            # --- Filter on group if requested ---
            filtered_indices = []
            if group_filter is None:
                filtered_indices = list(range(len(entries)))
            else:
                for idx, (_, grp, _) in enumerate(entries):
                    if grp is group_filter:
                        filtered_indices.append(idx)
            if not filtered_indices:
                continue

            # --- First device context we see (all bucket grads share it) ---
            device = self._local_grad(entries[0][0]).device

            # Precompute group → index map for tensor-LR/WD lookup (graph
            # region only; eager path still uses Python scalars so this
            # is just an id→int dict).
            group_to_idx = {id(g): i for i, g in enumerate(self.param_groups)}

            # --- Per-param pre-NS pass: momentum, tile, and record info
            #     needed to reconstruct update shape.
            # We build a parallel list ``per_param_state`` in _entries
            # order so indices align with ``tile_count_per_param`` /
            # ``param_offsets``.
            param_offsets: list[int] = []
            per_param_state: list[dict] = []
            running_offset = 0
            for idx, ((p, grp, tc), tile_count) in enumerate(
                zip(entries, tile_count_per_param)
            ):
                param_offsets.append(running_offset)
                running_offset += tile_count
                per_param_state.append(None)  # placeholder; filled below

            assert running_offset == B_total, (
                f"bucket {key}: sum(tile_count) {running_offset} != B_total {B_total}"
            )

            # Ensure concat buffer is at least (B_total, tile_h, tile_w).
            buf = self._ensure_concat_buffer(
                key, B_total, tile_h, tile_w, device, _dtype
            )

            for idx in filtered_indices:
                p, grp, tc = entries[idx]

                grad_eff = self._local_grad(p)
                g = self._compute_momentum(
                    p, grad_eff, grp["momentum"], grp["nesterov"]
                )

                h, w = g.shape
                pad_h = (tile_h - h % tile_h) % tile_h
                pad_w = (tile_w - w % tile_w) % tile_w
                H, W = h + pad_h, w + pad_w
                R, C = H // tile_h, W // tile_w
                Th, Tw = tile_h, tile_w
                info = (h, w, pad_h, pad_w)
                assert R * C == tc, f"tile count mismatch: planned {tc}, saw {R * C}"

                if pad_h or pad_w:
                    g_pad = F.pad(g, (0, pad_w, 0, pad_h))
                else:
                    g_pad = g
                ofs = param_offsets[idx]
                buf_4d = buf[ofs : ofs + tc].view(R, C, Th, Tw)
                buf_4d.copy_(g_pad.view(R, Th, C, Tw).permute(0, 2, 1, 3))

                per_param_state[idx] = {
                    "R": R,
                    "C": C,
                    "Th": Th,
                    "Tw": Tw,
                    "info": info,
                    "lr": grp["lr"],
                    "weight_decay": grp["weight_decay"],
                    "group_idx": group_to_idx[id(grp)],
                    "target_dtype": g.dtype,
                }

            # --- Batched NS: walk chunks, fire one NS call per chunk.
            #     NS output is kept as a new tensor per chunk — NOT copied
            #     back into the concat buffer. Scatter below slices the
            #     outputs directly, saving a step-level GPU memcpy.
            #
            #     CUDA event timing: record events per chunk but do NOT
            #     synchronize in the hot loop — sync once at the end.
            #     Per-chunk sync would serialize CPU/GPU and lose any
            #     submission overlap.
            ns_outputs: list[tuple[int, int, torch.Tensor]] = []
            ev_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
            ofs = 0
            for c in chunk_sizes:
                chunk = buf[ofs : ofs + c]  # (c, Th, Tw) view of concat buffer
                if use_events:
                    ev0 = torch.cuda.Event(enable_timing=True)
                    ev1 = torch.cuda.Event(enable_timing=True)
                    ev0.record()
                out = self.newton_schulz(chunk, steps=ns_steps)
                if use_events:
                    ev1.record()
                    ev_pairs.append((ev0, ev1))
                ns_outputs.append((ofs, c, out))
                ns_launches += 1
                total_B += c
                ofs += c

            # --- Per-param post-NS pass: locate each param's tiles in the
            #     NS output list (its tiles occupy [param_ofs,
            #     param_ofs+tc) in the concat buffer, which spans some
            #     contiguous range of chunks), gather, reshape, scale-
            #     divide, untile, WD, ``p.data.add_``.
            for idx in filtered_indices:
                st = per_param_state[idx]
                p, grp, _tc = entries[idx]
                R, C, Th, Tw = st["R"], st["C"], st["Th"], st["Tw"]
                tile_count = R * C
                param_start = param_offsets[idx]
                param_end = param_start + tile_count

                # Collect slices of ns_outputs that overlap this param's
                # [param_start, param_end). Most params fit in a single
                # chunk when chunks are memory-bounded (big), so this is
                # usually a single view. Only boundary-crossing params hit
                # the torch.cat path.
                pieces = []
                for chunk_ofs, chunk_size, out_tensor in ns_outputs:
                    overlap_start = max(param_start, chunk_ofs)
                    overlap_end = min(param_end, chunk_ofs + chunk_size)
                    if overlap_start < overlap_end:
                        local_start = overlap_start - chunk_ofs
                        local_end = overlap_end - chunk_ofs
                        pieces.append(out_tensor[local_start:local_end])
                    if chunk_ofs >= param_end:
                        break
                orth_flat = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)

                h_orig, w_orig, _ph, _pw = st["info"]
                update = self._post_ns_per_param(
                    orth_flat,
                    R,
                    C,
                    Th,
                    Tw,
                    h_orig,
                    w_orig,
                    st["target_dtype"],
                )

                self._apply_wd_and_update(
                    p, update, st["weight_decay"], st["lr"], st["group_idx"], (Th, Tw)
                )

            # --- Drain NS output tensors + sum CUDA event timings once.
            if use_events and ev_pairs:
                ev_pairs[-1][1].synchronize()  # one sync for the whole bucket
                for ev0, ev1 in ev_pairs:
                    ns_gpu_ms += ev0.elapsed_time(ev1)

        if ns_launches > 0:
            self._last_step_stats = {
                "ns_gpu_ms": float(ns_gpu_ms),
                "ns_launches": int(ns_launches),
                "total_B": int(total_B),
                "avg_batch": float(total_B) / float(ns_launches),
            }
        else:
            self._last_step_stats = {
                "ns_gpu_ms": 0.0,
                "ns_launches": 0,
                "total_B": 0,
                "avg_batch": 0.0,
            }

    def _step_muon_full_ns(self):
        """Per-param path for full-NS (single-tile) params.

        DTensor params are routed through the bucketed path by
        ``_plan_buckets`` and skipped here.
        """
        ns_gpu_ms = 0.0
        ns_launches = 0
        total_B = 0
        # Skip event timing inside graph capture (synchronize is illegal).
        use_events = torch.cuda.is_available() and not self._in_graph_region

        for group_idx, group in enumerate(self.param_groups):
            if not group["use_muon"]:
                continue
            tile_h, tile_w = group["tile_size"]
            ns_steps = group["ns_steps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if self._is_dtensor(p):
                    # Routed through the bucketed path.
                    continue
                full_ns = p.grad.numel() <= tile_h * tile_w
                if not full_ns:
                    continue  # bucketed path handles this one

                g = self._compute_momentum(
                    p, p.grad, group["momentum"], group["nesterov"]
                )

                if use_events:
                    ev0 = torch.cuda.Event(enable_timing=True)
                    ev1 = torch.cuda.Event(enable_timing=True)
                    ev0.record()
                update = self.newton_schulz(g, steps=ns_steps)
                if use_events:
                    ev1.record()
                    ev1.synchronize()
                    ns_gpu_ms += ev0.elapsed_time(ev1)
                ns_launches += 1
                total_B += 1

                self._apply_wd_and_update(
                    p, update, weight_decay, lr, group_idx, p.shape
                )

        return ns_gpu_ms, ns_launches, total_B

    def _step_adamw(self, group, group_idx):
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            grad_eff = self._local_grad(p)
            update = self._compute_adamw_step(
                p, grad_eff, group["adamw_betas"], group["adamw_eps"]
            )
            self._apply_wd_and_update(p, update, weight_decay, lr, group_idx)

    # -- Step --------------------------------------------------------------

    # -- Plan B: per-bank in-place path for FSDP DTensor ------------------

    def _step_muon_per_bank_inplace(self, p, group, group_idx: int) -> None:
        """Per-bank step for FSDP DTensor params: momentum → tile (optional)
        → batched NS → untile → WD + LR add. Runs entirely
        on the rank-local shard. Bypasses the bucket/concat infrastructure
        used in the non-FSDP path; that infrastructure leaves a permanent
        concat buffer that fragments the FSDP all-gather caching allocator
        when free GPU memory headroom is tight.
        """
        g_local = p.grad.to_local()
        if g_local.ndim != 3:
            raise RuntimeError(
                f"HiMuon FSDP per-bank path requires bank-wrapped 3D muon "
                f"params; got local ndim={g_local.ndim} (shape "
                f"{tuple(g_local.shape)}). Call wrap_with_banks() before "
                f"fully_shard()."
            )
        N, H, W = g_local.shape

        tile_h, tile_w = group["tile_size"]
        ns_steps = group["ns_steps"]

        g = self._compute_momentum(p, g_local, group["momentum"], group["nesterov"])

        if tile_h >= H and tile_w >= W:
            # Whole-matrix path: NS directly on (N, H, W). Equivalent to
            # MuonFsdp; R=C=1 makes the tiled path degenerate to NS's
            # spectral-norm step.
            u = self.newton_schulz(g, steps=ns_steps)
            block = (H, W)
        else:
            # Tiled path: per-matrix tile, batched NS over (N·R·C, T_h, T_w).
            pad_h = (tile_h - H % tile_h) % tile_h
            pad_w = (tile_w - W % tile_w) % tile_w
            if pad_h or pad_w:
                g_padded = F.pad(g, (0, pad_w, 0, pad_h))
            else:
                g_padded = g
            H_p, W_p = g_padded.shape[-2:]
            R, C = H_p // tile_h, W_p // tile_w
            tiled = (
                g_padded.view(N, R, tile_h, C, tile_w)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
            )

            flat = tiled.reshape(N * R * C, tile_h, tile_w)
            u_flat = self.newton_schulz(flat, steps=ns_steps)
            orth = u_flat.reshape(N, R, C, tile_h, tile_w)

            out = (
                orth.permute(0, 1, 3, 2, 4).contiguous().view(N, R * tile_h, C * tile_w)
            )
            if pad_h or pad_w:
                out = out[:, :H, :W]
            u = out
            block = (tile_h, tile_w)

        # Apply WD + LR update on the local shard (in-place).
        target = p.to_local()
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        if self._in_graph_region:
            scale = self.adjust_lr_for_muon(1.0, block)
            self._pending_updates.append(
                (p, u if scale == 1.0 else u.mul_(scale), group_idx)
            )
        else:
            if weight_decay > 0.0:
                target.mul_(1 - lr * weight_decay)
            adjusted_lr = self.adjust_lr_for_muon(lr, block)
            target.add_(u, alpha=-adjusted_lr)

    # -- Step body ---------------------------------------------------------

    def _has_dtensor_muon(self) -> bool:
        """True if any active muon param is a DTensor (FSDP)."""
        for g in self.param_groups:
            if not g["use_muon"]:
                continue
            for p in g["params"]:
                if p.grad is not None and self._is_dtensor(p):
                    return True
        return False

    def _step_body(self):
        """Side-effect only body of step(). Updates param_groups params
        in place. No closure handling; no return value. This is what
        the CUDA graph captures.

        Two paths:
          * **FSDP path (any DTensor muon params present)**: per-bank
            in-place via ``_step_muon_per_bank_inplace``. No concat
            buffer, no plan/bucket cache, no post-NS scatter — eliminates
            the permanent footprint that fragments the caching allocator
            when GPU headroom is tight.
          * **Non-FSDP path (no DTensor params)**: cross-layer concat
            buckets via ``_plan_buckets`` + ``_execute_plan`` + per-param
            full-NS fallback. This is the original HiMuon behaviour for
            single-GPU and DDP.
        """
        if self._has_dtensor_muon():
            # FSDP path
            ns_launches = 0
            for group_idx, group in enumerate(self.param_groups):
                if group["use_muon"]:
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        if not self._is_dtensor(p):
                            raise RuntimeError(
                                "HiMuon FSDP path: muon param group contains a "
                                "non-DTensor param. Mixed DTensor/non-DTensor "
                                "muon params are not supported. Wrap all "
                                "muon-eligible params with the same FSDP "
                                "scheme."
                            )
                        self._step_muon_per_bank_inplace(p, group, group_idx)
                        ns_launches += 1
                else:
                    self._step_adamw(group, group_idx)
            self._last_step_stats = {
                "ns_gpu_ms": 0.0,
                "ns_launches": int(ns_launches),
                "total_B": 0,
                "avg_batch": 0.0,
            }
            return

        # Non-FSDP path (original cross-layer concat).
        if self._cached_plan is None:
            self._cached_plan = self._plan_buckets()
        plan = self._cached_plan

        self._execute_plan(plan)
        bucketed = dict(self._last_step_stats)

        pp_gpu_ms, pp_launches, pp_B = self._step_muon_full_ns()

        for group_idx, group in enumerate(self.param_groups):
            if not group["use_muon"]:
                self._step_adamw(group, group_idx)

        total_launches = bucketed["ns_launches"] + pp_launches
        total_B = bucketed["total_B"] + pp_B
        total_ms = bucketed["ns_gpu_ms"] + pp_gpu_ms
        self._last_step_stats = {
            "ns_gpu_ms": float(total_ms),
            "ns_launches": int(total_launches),
            "total_B": int(total_B),
            "avg_batch": (
                float(total_B) / float(total_launches) if total_launches > 0 else 0.0
            ),
        }

    def _capture_graph(self) -> None:
        """Capture the update-compute part of ``step()`` into
        ``self._cuda_graph``. The captured region covers momentum,
        tile, NS, and scatter (producing the per-param ``update``
        tensors) but STOPS before WD/LR application. Populates
        ``self._pending_updates`` as a Python-side list of
        ``(param, update_tensor, group_idx)``; replay refreshes the
        update tensor contents, and ``_apply_pending_updates`` applies
        the final WD + LR*update with scalar reads from
        ``group["lr"]`` / ``group["weight_decay"]``.
        """
        torch.cuda.synchronize()
        self._pending_updates.clear()
        self._cuda_graph = torch.cuda.CUDAGraph()
        self._in_graph_region = True
        try:
            with torch.cuda.graph(self._cuda_graph):
                self._step_body()
        finally:
            self._in_graph_region = False
        # Replay once + apply so this step() produces the same update as
        # eager would (user expects step() to always step).
        self._cuda_graph.replay()
        self._apply_pending_updates()
        torch.cuda.synchronize()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if not self._cuda_graph_enabled or not torch.cuda.is_available():
            self._step_body()
            return loss

        # Eager warmup phase: let lazy state init (momentum_buffer),
        # concat buffers, Triton autotune, and caching allocator all
        # stabilise before capture.
        if self._graph_step_count < self._cuda_graph_warmup:
            self._step_body()
            self._graph_step_count += 1
            return loss

        # Capture phase: first post-warmup step.
        if self._cuda_graph is None:
            self._capture_graph()
            return loss

        # Steady state: replay then apply WD+LR outside the graph.
        # Reading group["lr"] / group["weight_decay"] inside
        # ``_apply_pending_updates`` gives LR schedulers a free ride
        # — no recapture, no tensor-pointer syncing.
        self._cuda_graph.replay()
        self._apply_pending_updates()
        self._last_step_stats = {
            "ns_gpu_ms": 0.0,
            "ns_launches": 0,
            "total_B": 0,
            "avg_batch": 0.0,
        }
        return loss
