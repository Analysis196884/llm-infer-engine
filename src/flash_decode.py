import math

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_decode_split_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    mask_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    partial_acc_ptr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kl,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vl,
    stride_vd,
    stride_mask_b,
    stride_mask_k,
    stride_pmb,
    stride_pmh,
    stride_pms,
    stride_psb,
    stride_psh,
    stride_pss,
    stride_pab,
    stride_pah,
    stride_pas,
    stride_pad,
    n_heads,
    n_kv_heads,
    kv_len,
    sm_scale: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    SPLIT_KV: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    split_id = tl.program_id(2)
    kv_head_id = head_id // (n_heads // n_kv_heads)

    d = tl.arange(0, BLOCK_D)
    q = tl.load(
        q_ptr
        + batch_id * stride_qb
        + head_id * stride_qh
        + d * stride_qd,
        mask=d < HEAD_DIM,
        other=0.0,
    ).to(tl.float32)

    k_base = k_ptr + batch_id * stride_kb + kv_head_id * stride_kh
    v_base = v_ptr + batch_id * stride_vb + kv_head_id * stride_vh
    split_start = split_id * SPLIT_KV

    m_i = tl.full((1,), float("-inf"), tl.float32)
    l_i = tl.zeros((1,), tl.float32)
    acc = tl.zeros((BLOCK_D,), tl.float32)

    for block_start in range(0, SPLIT_KV, BLOCK_KV):
        positions = split_start + block_start + tl.arange(0, BLOCK_KV)
        valid = positions < kv_len

        k = tl.load(
            k_base
            + positions[:, None] * stride_kl
            + d[None, :] * stride_kd,
            mask=valid[:, None] & (d[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * sm_scale
        scores = tl.where(valid, scores, float("-inf"))

        if HAS_MASK:
            keep = tl.load(
                mask_ptr
                + batch_id * stride_mask_b
                + positions * stride_mask_k,
                mask=valid,
                other=False,
            )
            scores = tl.where(keep, scores, float("-inf"))

        m_block = tl.max(scores, axis=0)
        # Keep an empty block numerically inert instead of evaluating -inf - -inf.
        m_block = tl.maximum(m_block, -1.0e9)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        probabilities = tl.exp(scores - m_new)

        v = tl.load(
            v_base
            + positions[:, None] * stride_vl
            + d[None, :] * stride_vd,
            mask=valid[:, None] & (d[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        acc = acc * alpha + tl.sum(probabilities[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(probabilities, axis=0)
        m_i = m_new

    max_offset = (
        batch_id * stride_pmb
        + head_id * stride_pmh
        + split_id * stride_pms
    )
    sum_offset = (
        batch_id * stride_psb
        + head_id * stride_psh
        + split_id * stride_pss
    )
    acc_offset = (
        batch_id * stride_pab
        + head_id * stride_pah
        + split_id * stride_pas
    )
    singleton = tl.arange(0, 1)
    tl.store(partial_max_ptr + max_offset + singleton, m_i)
    tl.store(partial_sum_ptr + sum_offset + singleton, l_i)
    tl.store(
        partial_acc_ptr + acc_offset + d * stride_pad,
        acc,
        mask=d < HEAD_DIM,
    )


@triton.jit
def _flash_decode_reduce_kernel(
    partial_max_ptr,
    partial_sum_ptr,
    partial_acc_ptr,
    out_ptr,
    stride_pmb,
    stride_pmh,
    stride_pms,
    stride_psb,
    stride_psh,
    stride_pss,
    stride_pab,
    stride_pah,
    stride_pas,
    stride_pad,
    stride_ob,
    stride_oh,
    stride_od,
    n_heads,
    num_splits,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    splits = tl.arange(0, BLOCK_SPLITS)
    valid_split = splits < num_splits

    max_base = partial_max_ptr + batch_id * stride_pmb + head_id * stride_pmh
    sum_base = partial_sum_ptr + batch_id * stride_psb + head_id * stride_psh
    split_max = tl.load(
        max_base + splits * stride_pms,
        mask=valid_split,
        other=float("-inf"),
    )
    split_sum = tl.load(
        sum_base + splits * stride_pss,
        mask=valid_split,
        other=0.0,
    )
    global_max = tl.max(
        tl.where(split_sum > 0.0, split_max, float("-inf")),
        axis=0,
    )
    global_max = tl.maximum(global_max, -1.0e9)
    weights = tl.where(
        (split_sum > 0.0) & valid_split,
        tl.exp(split_max - global_max),
        0.0,
    )
    denominator = tl.sum(weights * split_sum, axis=0)

    d = tl.arange(0, BLOCK_D)
    acc_base = partial_acc_ptr + batch_id * stride_pab + head_id * stride_pah
    split_acc = tl.load(
        acc_base
        + splits[:, None] * stride_pas
        + d[None, :] * stride_pad,
        mask=valid_split[:, None] & (d[None, :] < HEAD_DIM),
        other=0.0,
    )
    numerator = tl.sum(weights[:, None] * split_acc, axis=0)
    safe_denominator = tl.maximum(denominator, 1.0)
    output = tl.where(denominator > 0.0, numerator / safe_denominator, 0.0)

    out_base = out_ptr + batch_id * stride_ob + head_id * stride_oh
    tl.store(
        out_base + d * stride_od,
        output,
        mask=d < HEAD_DIM,
    )


def _mask_strides(mask: torch.Tensor, batch_size: int, kv_len: int) -> tuple[int, int]:
    if mask.dtype != torch.bool:
        raise TypeError(f"mask must have dtype torch.bool, got {mask.dtype}")
    if mask.ndim == 0:
        raise ValueError("mask must have at least one dimension")
    if mask.ndim == 1:
        if mask.shape[0] < kv_len:
            raise ValueError("mask is shorter than the KV sequence")
        return 0, mask.stride(0)

    if mask.shape[0] not in (1, batch_size):
        raise ValueError(
            f"mask batch dimension must be 1 or {batch_size}, got {mask.shape[0]}"
        )
    if any(size != 1 for size in mask.shape[1:-1]):
        raise ValueError("all mask dimensions except batch and KV length must be 1")
    if mask.shape[-1] < kv_len:
        raise ValueError("mask is shorter than the KV sequence")
    return (0 if mask.shape[0] == 1 else mask.stride(0), mask.stride(-1))


def flash_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute single-token GQA/MQA attention with split-KV parallelism.

    Shapes are q=[B, Hq, 1, D] and k/v=[B, Hkv, L, D]. A boolean mask may
    have shape [L], [B, L], or [B, 1, 1, L].
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must all be rank-4 tensors")

    batch_size, n_heads, query_len, head_dim = q.shape
    k_batch, n_kv_heads, kv_len, k_head_dim = k.shape
    if query_len != 1:
        raise ValueError(f"flash_decode only supports query_len=1, got {query_len}")
    if k.shape != v.shape:
        raise ValueError(f"k and v shapes must match, got {k.shape} and {v.shape}")
    if k_batch != batch_size or k_head_dim != head_dim:
        raise ValueError("q and k/v batch size and head dimension must match")
    if kv_len == 0:
        raise ValueError("KV sequence must not be empty")
    if n_heads % n_kv_heads != 0:
        raise ValueError(
            f"query heads ({n_heads}) must be divisible by KV heads ({n_kv_heads})"
        )
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("flash_decode requires CUDA tensors")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("q, k, and v must have the same dtype")

    block_kv = 64
    split_kv = 256
    num_splits = triton.cdiv(kv_len, split_kv)
    block_d = triton.next_power_of_2(head_dim)
    block_splits = triton.next_power_of_2(num_splits)
    if block_d > 256:
        raise ValueError(f"head_dim must be <= 256, got {head_dim}")
    if block_splits > 256:
        raise ValueError(f"KV sequence is too long for flash_decode: {kv_len}")

    mask_stride_b = 0
    mask_stride_k = 0
    if mask is not None:
        if not mask.is_cuda or mask.device != q.device:
            raise ValueError("mask must be on the same CUDA device as q")
        mask_stride_b, mask_stride_k = _mask_strides(mask, batch_size, kv_len)
    mask_ptr = q if mask is None else mask

    partial_max = torch.empty(
        (batch_size, n_heads, num_splits), device=q.device, dtype=torch.float32
    )
    partial_sum = torch.empty_like(partial_max)
    partial_acc = torch.empty(
        (batch_size, n_heads, num_splits, head_dim),
        device=q.device,
        dtype=torch.float32,
    )
    out = torch.empty_like(q)

    split_grid = (batch_size, n_heads, num_splits)
    _flash_decode_split_kernel[split_grid](
        q,
        k,
        v,
        mask_ptr,
        partial_max,
        partial_sum,
        partial_acc,
        q.stride(0),
        q.stride(1),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        mask_stride_b,
        mask_stride_k,
        partial_max.stride(0),
        partial_max.stride(1),
        partial_max.stride(2),
        partial_sum.stride(0),
        partial_sum.stride(1),
        partial_sum.stride(2),
        partial_acc.stride(0),
        partial_acc.stride(1),
        partial_acc.stride(2),
        partial_acc.stride(3),
        n_heads,
        n_kv_heads,
        kv_len,
        sm_scale=1.0 / math.sqrt(head_dim),
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        BLOCK_KV=block_kv,
        SPLIT_KV=split_kv,
        HAS_MASK=mask is not None,
        num_warps=4,
    )

    reduce_grid = (batch_size, n_heads)
    _flash_decode_reduce_kernel[reduce_grid](
        partial_max,
        partial_sum,
        partial_acc,
        out,
        partial_max.stride(0),
        partial_max.stride(1),
        partial_max.stride(2),
        partial_sum.stride(0),
        partial_sum.stride(1),
        partial_sum.stride(2),
        partial_acc.stride(0),
        partial_acc.stride(1),
        partial_acc.stride(2),
        partial_acc.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(3),
        n_heads,
        num_splits,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        BLOCK_SPLITS=block_splits,
        num_warps=4,
    )
    return out


# Keep the attention-style name convenient for callers.
flash_decode_attention = flash_decode
