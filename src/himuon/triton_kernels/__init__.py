try:
    from .ba_plus_cAA import ba_plus_cAA
    from .fused_bmm_add import fused_bmm_add
    from .ns5_smem import ns5_smem, ns5_smem_kernel
    from .XXT import XXT

    _KERNELS_AVAILABLE = True
except Exception as _import_err:
    _KERNELS_AVAILABLE = False
    _REASON = _import_err

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "himuon Triton kernels are unavailable on this host "
            f"({type(_REASON).__name__}: {_REASON}). A GPU is required to run "
            "Optimizer.step(); the CPU-only contract test tier never calls it."
        )

    XXT = ba_plus_cAA = fused_bmm_add = ns5_smem = ns5_smem_kernel = _unavailable

__all__ = (
    "XXT",
    "ba_plus_cAA",
    "fused_bmm_add",
    "ns5_smem",
    "ns5_smem_kernel",
)
__version__ = "1.0"
