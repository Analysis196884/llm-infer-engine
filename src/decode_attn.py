import torch
import triton
import triton.language as tl


@triton.jit
def _decode_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, mask_ptr,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kl, stride_kd,
    stride_vb, stride_vh, stride_vl, stride_vd,
    stride_mask_b, stride_mask_k,
    n_heads, n_kv_heads, kv_loop_len, head_dim: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_id = pid // n_heads
    head_id = pid % n_heads
    kv_head_id = head_id // (n_heads // n_kv_heads)

    q_off = q_ptr + batch_id * stride_qb + head_id * stride_qh
    q = tl.load(q_off + tl.arange(0, head_dim) * stride_qd)

    k_base = k_ptr + batch_id * stride_kb + kv_head_id * stride_kh
    v_base = v_ptr + batch_id * stride_vb + kv_head_id * stride_vh

    m_i = tl.full([1], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([head_dim], dtype=tl.float32)

    kv_scale = 1.0 / (head_dim ** 0.5)
    cols = tl.arange(0, head_dim)
    rows = tl.arange(0, BLOCK_KV)

    for start in range(0, kv_loop_len, BLOCK_KV):
        offs = start + rows
        mask_bounds = offs < kv_loop_len

        k_ptrs = k_base + offs[:, None] * stride_kl + cols[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_bounds[:, None], other=0.0)

        scores = tl.sum(q.to(tl.float32) * k.to(tl.float32), axis=1) * kv_scale
        scores = tl.where(mask_bounds, scores, float("-inf"))

        if HAS_MASK:
            m_ptrs = mask_ptr + batch_id * stride_mask_b + offs * stride_mask_k
            m_val = tl.load(m_ptrs, mask=mask_bounds, other=False)
            scores = tl.where(m_val, scores, float("-inf"))

        m_block = tl.max(scores, 0)
        m_block = tl.maximum(m_block, -1.0e9)

        p_block = tl.exp(scores - m_block)
        l_block = tl.sum(p_block, 0)

        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_block - m_new)
        l_new = alpha * l_i + beta * l_block

        v_ptrs = v_base + offs[:, None] * stride_vl + cols[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_bounds[:, None], other=0.0)

        acc = acc * alpha + tl.sum(p_block[:, None] * v.to(tl.float32), axis=0) * beta

        m_i = m_new
        l_i = l_new

    out = acc / l_i
    out_off = out_ptr + batch_id * stride_qb + head_id * stride_qh
    tl.store(out_off + tl.arange(0, head_dim) * stride_qd, out)


@triton.jit
def _decode_attention_nomask_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kl, stride_kd,
    stride_vb, stride_vh, stride_vl, stride_vd,
    n_heads, n_kv_heads, kv_loop_len, head_dim: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_id = pid // n_heads
    head_id = pid % n_heads
    kv_head_id = head_id // (n_heads // n_kv_heads)

    q_off = q_ptr + batch_id * stride_qb + head_id * stride_qh
    q = tl.load(q_off + tl.arange(0, head_dim) * stride_qd)

    k_base = k_ptr + batch_id * stride_kb + kv_head_id * stride_kh
    v_base = v_ptr + batch_id * stride_vb + kv_head_id * stride_vh

    m_i = tl.full([1], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([head_dim], dtype=tl.float32)

    kv_scale = 1.0 / (head_dim ** 0.5)
    cols = tl.arange(0, head_dim)
    rows = tl.arange(0, BLOCK_KV)

    for start in range(0, kv_loop_len, BLOCK_KV):
        offs = start + rows
        mask_bounds = offs < kv_loop_len

        k_ptrs = k_base + offs[:, None] * stride_kl + cols[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_bounds[:, None], other=0.0)

        scores = tl.sum(q.to(tl.float32) * k.to(tl.float32), axis=1) * kv_scale
        scores = tl.where(mask_bounds, scores, float("-inf"))

        m_block = tl.max(scores, 0)
        m_block = tl.maximum(m_block, -1.0e9)

        p_block = tl.exp(scores - m_block)
        l_block = tl.sum(p_block, 0)

        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_block - m_new)
        l_new = alpha * l_i + beta * l_block

        v_ptrs = v_base + offs[:, None] * stride_vl + cols[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_bounds[:, None], other=0.0)

        acc = acc * alpha + tl.sum(p_block[:, None] * v.to(tl.float32), axis=0) * beta

        m_i = m_new
        l_i = l_new

    out = acc / l_i
    out_off = out_ptr + batch_id * stride_qb + head_id * stride_qh
    tl.store(out_off + tl.arange(0, head_dim) * stride_qd, out)


def decode_attention(xq, xk, xv, mask=None):
    B, n_heads, q_len, head_dim = xq.shape
    _, n_kv_heads, kv_loop_len, _ = xk.shape

    out = torch.empty_like(xq)
    grid = (B * n_heads,)
    BLOCK_KV = 64

    if mask is not None:
        _decode_attention_kernel[grid](
            xq, xk, xv, out, mask,
            xq.stride(0), xq.stride(1), xq.stride(3),
            xk.stride(0), xk.stride(1), xk.stride(2), xk.stride(3),
            xv.stride(0), xv.stride(1), xv.stride(2), xv.stride(3),
            mask.stride(0), mask.stride(-1),
            n_heads, n_kv_heads, kv_loop_len, head_dim,
            BLOCK_KV=BLOCK_KV,
            HAS_MASK=True,
        )
    else:
        _decode_attention_nomask_kernel[grid](
            xq, xk, xv, out,
            xq.stride(0), xq.stride(1), xq.stride(3),
            xk.stride(0), xk.stride(1), xk.stride(2), xk.stride(3),
            xv.stride(0), xv.stride(1), xv.stride(2), xv.stride(3),
            n_heads, n_kv_heads, kv_loop_len, head_dim,
            BLOCK_KV=BLOCK_KV,
        )

    return out
