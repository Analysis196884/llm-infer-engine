import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def w8a16_gemv_kernel(
    x_ptr,
    w_ptr,
    scale_ptr,
    out_ptr,
    M: tl.constexpr, # M = 1 or M small enough to benefit from GEMV
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k

        x = tl.load(
            x_ptr + pid_m * stride_xm + k_idxs * stride_xk,
            mask=k_idxs < K,
            other=0.0,
        ).to(tl.float32)

        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + k_idxs[None, :] * stride_wk,
            mask=(offs_n[:, None] < N) & (k_idxs[None, :] < K),
            other=0,
        ).to(tl.float32)

        acc += tl.sum(w * x[None, :], axis=1)

    scale = tl.load(
        scale_ptr + offs_n * stride_sn,
        mask=offs_n < N,
        other=0.0,
    ).to(tl.float32)

    acc = acc * scale

    tl.store(
        out_ptr + pid_m * stride_om + offs_n * stride_on,
        acc,
        mask=offs_n < N,
    )


def w8a16_gemv_launch(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    *,
    block_n: int = 32,
    block_k: int = 256,
) -> torch.Tensor:
    """
    x:      [M, K], bf16/fp16
    weight: [N, K], int8
    scales: [N, 1] or [N], bf16/fp16/fp32
    out:    [M, N], same dtype as x
    """
    assert x.dim() == 2
    assert weight.dim() == 2
    assert weight.dtype == torch.int8
    assert x.is_cuda and weight.is_cuda and scales.is_cuda
    assert x.shape[1] == weight.shape[1]

    M, K = x.shape
    N = weight.shape[0]

    if scales.dim() == 2:
        assert scales.shape == (N, 1)
        scales_1d = scales.view(N)
    else:
        assert scales.shape == (N,)
        scales_1d = scales

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = (triton.cdiv(N, block_n), M)

    w8a16_gemv_kernel[grid](
        x,
        weight,
        scales_1d,
        out,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        scales_1d.stride(0),
        out.stride(0),
        out.stride(1),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )

    return out


@triton.jit
def w8a16_gemm_kernel(
    x_ptr,
    w_ptr,
    scale_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_sn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """C = x @ w.T * scale, where x is bf16/fp16 and w is int8.

    Shapes:
        x:     [M, K]
        w:     [N, K] (int8)
        scale: [N, 1] or [N]
        out:   [M, N]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k

        x_block = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k_idxs[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
            other=0.0,
        )
        w_block = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + k_idxs[None, :] * stride_wk,
            mask=(offs_n[:, None] < N) & (k_idxs[None, :] < K),
            other=0,
        )
        w_block = w_block.to(x_block.dtype)

        accumulator += tl.dot(x_block, tl.trans(w_block), out_dtype=tl.float32)

    scale = tl.load(
        scale_ptr + offs_n * stride_sn,
        mask=offs_n < N,
        other=0.0,
    ).to(tl.float32)
    accumulator = accumulator * scale[None, :]

    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        accumulator,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def w8a16_gemm_launch(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    *,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 128,
) -> torch.Tensor:
    """
    x:      [M, K], bf16/fp16
    weight: [N, K], int8
    scales: [N, 1] or [N], bf16/fp16/fp32
    out:    [M, N], same dtype as x
    """
    assert x.dim() == 2
    assert weight.dim() == 2
    assert weight.dtype == torch.int8
    assert x.is_cuda and weight.is_cuda and scales.is_cuda
    assert x.shape[1] == weight.shape[1]

    M, K = x.shape
    N = weight.shape[0]

    if scales.dim() == 2:
        assert scales.shape == (N, 1)
        scales_1d = scales.view(N)
    else:
        assert scales.shape == (N,)
        scales_1d = scales

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    w8a16_gemm_kernel[grid](
        x,
        weight,
        scales_1d,
        out,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        scales_1d.stride(0),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )

    return out


class W8A16Linear(nn.Module):
    """Per-channel symmetric int8 weight, bf16/fp16 activation linear layer."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        if bias:
            raise NotImplementedError("W8A16Linear does not support bias yet.")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.int8),
            requires_grad=False,
        )
        self.scales = nn.Parameter(
            torch.empty(out_features, 1, dtype=torch.bfloat16),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)

        # decode/small-M path
        if x_2d.shape[0] <= 8:
            out_2d = w8a16_gemv_launch(
                x_2d,
                self.weight,
                self.scales,
                block_n=32,
                block_k=256,
            )
        else:
            # prefill path: block GEMM
            out_2d = w8a16_gemm_launch(
                x_2d,
                self.weight,
                self.scales,
                block_m=64,
                block_n=64,
                block_k=128,
            )

        return out_2d.reshape(*orig_shape[:-1], self.out_features)