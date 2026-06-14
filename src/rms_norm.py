import torch
import torch.nn as nn
import triton
import triton.language as tl

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        torch.cuda.nvtx.range_push("RMSNorm")
        out = rmsnorm_triton(x, self.weight, self.eps)
        torch.cuda.nvtx.range_pop()
        return out

@triton.jit
def rms_norm_fwd_kernel(
    X,          # [M, N]
    W,          # [N]
    Y,          # [M, N]
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)

    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)

    ss = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(ss + eps)

    y = x * rstd * w
    tl.store(Y + row * N + offs, y, mask=mask)


def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    assert x.is_cuda
    assert weight.is_cuda
    assert x.shape[-1] == weight.numel()

    orig_shape = x.shape
    N = x.shape[-1]
    x_2d = x.contiguous().view(-1, N)
    M = x_2d.shape[0]

    y = torch.empty_like(x_2d)

    BLOCK = triton.next_power_of_2(N)

    # hidden dim 常见 1024/2048/4096/8192/16384
    # BLOCK 过大时寄存器压力会很高，需要拆分 reduction 或用更复杂实现
    assert BLOCK <= 65536

    if BLOCK <= 1024:
        num_warps = 4
    elif BLOCK <= 4096:
        num_warps = 8
    else:
        num_warps = 16

    rms_norm_fwd_kernel[(M,)](
        x_2d,
        weight,
        y,
        N,
        eps,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )

    return y.view(orig_shape)