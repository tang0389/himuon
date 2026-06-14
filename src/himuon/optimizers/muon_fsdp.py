"""
Muon adapted for FSDP2 with parameter-bank sharding. Mirrors
``muon.py`` line-by-line except for the DTensor plumbing needed to
read from / write to each rank's local bank shard.
"""

import torch
from torch import Tensor
import math

try:
    from torch.distributed.tensor import DTensor as _DTensor  # type: ignore
except Exception:  # pragma: no cover
    _DTensor = None  # type: ignore


# -----------------------------------------------------------------------------
# Muon optimizer
# -----------------------------------------------------------------------------
@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert (
        G.ndim >= 2
    )  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class MuonFsdp(torch.optim.Optimizer):
    """
    Muon (FSDP2 bank-sharded variant). See ``Muon`` for the algorithm.
    Bank DTensor params route NS through the local shard; each rank holds
    complete (H, W) matrices on the layer axis, so per-matrix NS needs
    no cross-rank collectives.
    """

    def __init__(
        self,
        params,
        lr=0.02,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        use_muon=True,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            use_muon=use_muon,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        super().__init__(params, defaults)

    @staticmethod
    def _is_dtensor(p) -> bool:
        return _DTensor is not None and isinstance(p, _DTensor)

    @classmethod
    def _local(cls, p):
        """Return the local tensor backing ``p``.

        Uses ``p.data.to_local()`` rather than ``p.to_local()`` so the
        returned tensor bypasses DTensor's autograd view and can be
        safely mutated in-place during the optimizer step.
        """
        if cls._is_dtensor(p):
            return p.data.to_local()
        return p.data

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if group["use_muon"]]
            # import pdb; pdb.set_trace()
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            # generate weight updates in distributed fashion
            for p in params:
                # sanity check
                if p.grad is None:
                    continue
                g = self._local(p.grad)
                if g.ndim > 2 and not self._is_dtensor(p):
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                # scale update — for 3D bank shard (N, H, W) the per-matrix
                # dims live in the last two axes; otherwise use p.shape.
                matrix_shape = g.shape[-2:] if self._is_dtensor(p) else p.shape
                adjusted_lr = self.adjust_lr_for_muon(lr, matrix_shape)

                # apply weight decay + update on the local shard
                target = self._local(p)
                target.mul_(1 - lr * weight_decay)
                target.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not group["use_muon"]]
            lr = group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["weight_decay"]

            for p in params:
                if p.grad is None:
                    continue
                g = self._local(p.grad)
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                target = self._local(p)
                target.mul_(1 - lr * weight_decay)
                target.add_(g, alpha=-lr / scale)

        return loss
