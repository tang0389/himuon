import triton
import triton.language as tl


def get_polar_autotune_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": 8,
                "LOWER_UPPER": 1,
            },
            num_stages=stages,
            num_warps=warps,
        )
        for bm in [64, 128]
        for bn in [64, 128, 256]
        for bk in [64, 128]
        for stages, warps in [(3, 4), (3, 8), (4, 4)]
        if bm // bn <= 2 and bn // bm <= 2
    ]


def get_bmm_autotune_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": m,
                "BLOCK_SIZE_N": n,
                "BLOCK_SIZE_K": k,
                "GROUP_SIZE_M": 8,
            },
            num_stages=s,
            num_warps=w,
        )
        for m, n, k, s, w in [
            (128, 256, 64, 3, 8),
            (64, 256, 32, 4, 4),
            (128, 128, 32, 4, 4),
            (128, 64, 32, 4, 4),
            (64, 128, 32, 4, 4),
            (128, 32, 32, 4, 4),
            (64, 32, 32, 5, 2),
            (32, 64, 32, 5, 2),
        ]
    ]


@triton.jit
def polar_pid_to_block(
    pid,
    M,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Split output matrix into blocks of size (BLOCK_SIZE_M, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)

    # Map PID to a single matrix in batch
    batch_idx = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)

    # Map PID to 2D grid of blocks
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    m_idx = pid_m * BLOCK_SIZE_M
    n_idx = pid_n * BLOCK_SIZE_N
    return batch_idx, m_idx, n_idx
