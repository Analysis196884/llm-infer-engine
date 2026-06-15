import math

import torch
import triton
import triton.language as tl


def _compute_inv_freq(dim: int, theta: float, rope_scaling=None):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    if not rope_scaling or rope_scaling.get("rope_type") != "llama3":
        return inv_freq

    factor = float(rope_scaling["factor"])
    low_freq_factor = float(rope_scaling["low_freq_factor"])
    high_freq_factor = float(rope_scaling["high_freq_factor"])
    old_context_len = float(rope_scaling["original_max_position_embeddings"])

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)

    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = (~(wavelen < high_freq_wavelen)) & (~(wavelen > low_freq_wavelen))
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama


def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0, rope_scaling=None):
    freqs = _compute_inv_freq(dim=dim, theta=theta, rope_scaling=rope_scaling)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def build_rope_cos_sin(freqs_cis, dtype, device):
    """Return half-length cos/sin of shape [seq_len, head_dim // 2]."""
    cos_half = freqs_cis.real.to(device=device, dtype=dtype)
    sin_half = freqs_cis.imag.to(device=device, dtype=dtype)
    return cos_half, sin_half


@triton.jit
def rope_kernel(
    x_ptr, out_ptr, cos_ptr, sin_ptr,
    B, n_heads, seq_len,
    stride_xb, stride_xh, stride_xs, stride_xd,
    stride_cs, stride_cd,
    head_dim: tl.constexpr,
    half_dim: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)

    batch_id = pid_bh // n_heads
    head_id = pid_bh % n_heads
    seq_start = pid_s * BLOCK_SEQ

    rows = tl.arange(0, BLOCK_SEQ)
    cols = tl.arange(0, half_dim)
    seq_offs = seq_start + rows

    mask_seq = seq_offs < seq_len

    x_base = x_ptr + batch_id * stride_xb + head_id * stride_xh
    out_base = out_ptr + batch_id * stride_xb + head_id * stride_xh

    x1_ptrs = x_base + seq_offs[:, None] * stride_xs + cols[None, :] * stride_xd
    x1 = tl.load(x1_ptrs, mask=mask_seq[:, None], other=0.0)

    x2_ptrs = x_base + seq_offs[:, None] * stride_xs + (cols + half_dim)[None, :] * stride_xd
    x2 = tl.load(x2_ptrs, mask=mask_seq[:, None], other=0.0)

    cos_ptrs = cos_ptr + seq_offs[:, None] * stride_cs + cols[None, :] * stride_cd
    sin_ptrs = sin_ptr + seq_offs[:, None] * stride_cs + cols[None, :] * stride_cd
    c = tl.load(cos_ptrs, mask=mask_seq[:, None], other=0.0)
    s = tl.load(sin_ptrs, mask=mask_seq[:, None], other=0.0)

    y1 = x1 * c - x2 * s
    y2 = x2 * c + x1 * s

    tl.store(out_base + seq_offs[:, None] * stride_xs + cols[None, :] * stride_xd,
             y1, mask=mask_seq[:, None])
    tl.store(out_base + seq_offs[:, None] * stride_xs + (cols + half_dim)[None, :] * stride_xd,
             y2, mask=mask_seq[:, None])


def apply_rotary(x, cos, sin):
    """
    Args:
        x:      [B, n_heads, seq_len, head_dim]
        cos:    [seq_len, head_dim // 2]
        sin:    [seq_len, head_dim // 2]
    Returns:
        out:    [B, n_heads, seq_len, head_dim]
    """
    B, n_heads, seq_len, head_dim = x.shape
    out = torch.empty_like(x)

    BLOCK_SEQ = 1 if seq_len == 1 else 32
    grid = (B * n_heads, (seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ)

    rope_kernel[grid](
        x, out, cos, sin,
        B, n_heads, seq_len,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1),
        head_dim=head_dim,
        half_dim=head_dim // 2,
        BLOCK_SEQ=BLOCK_SEQ,
    )

    return out


@triton.jit
def _rope_k_cache_write_seq_kernel(
    xk_ptr, xv_ptr, cos_ptr, sin_ptr,
    cache_k_ptr, cache_v_ptr,
    B, n_kv_heads, input_len, head_dim,
    stride_xkb, stride_xkh, stride_xks, stride_xkd,
    stride_xvb, stride_xvh, stride_xvs, stride_xvd,
    stride_ckb, stride_ckh, stride_cks, stride_ckd,
    stride_cvb, stride_cvh, stride_cvs, stride_cvd,
    stride_cs, stride_cd,
    start_pos,
    BLOCK_SEQ: tl.constexpr,
    half_dim: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)

    batch_id = pid_bh // n_kv_heads
    head_id = pid_bh % n_kv_heads
    seq_start = pid_s * BLOCK_SEQ

    rows = tl.arange(0, BLOCK_SEQ)
    cols = tl.arange(0, half_dim)
    seq_offs = seq_start + rows

    mask_seq = seq_offs < input_len

    # --- xk RoPE ---
    xk_base = xk_ptr + batch_id * stride_xkb + head_id * stride_xkh
    x1_ptrs = xk_base + seq_offs[:, None] * stride_xks + cols[None, :] * stride_xkd
    x2_ptrs = xk_base + seq_offs[:, None] * stride_xks + (cols + half_dim)[None, :] * stride_xkd
    x1 = tl.load(x1_ptrs, mask=mask_seq[:, None], other=0.0)
    x2 = tl.load(x2_ptrs, mask=mask_seq[:, None], other=0.0)

    # cos/sin indexed relative to sliced array start
    cos_ptrs = cos_ptr + seq_offs[:, None] * stride_cs + cols[None, :] * stride_cd
    sin_ptrs = sin_ptr + seq_offs[:, None] * stride_cs + cols[None, :] * stride_cd
    c = tl.load(cos_ptrs, mask=mask_seq[:, None], other=0.0)
    s = tl.load(sin_ptrs, mask=mask_seq[:, None], other=0.0)

    y1 = x1 * c - x2 * s
    y2 = x2 * c + x1 * s

    # write RoPE'd xk into cache_k at start_pos + seq_offs
    abs_pos = start_pos + seq_offs
    cache_k_base = cache_k_ptr + batch_id * stride_ckb + head_id * stride_ckh
    ck1_ptrs = cache_k_base + abs_pos[:, None] * stride_cks + cols[None, :] * stride_ckd
    ck2_ptrs = cache_k_base + abs_pos[:, None] * stride_cks + (cols + half_dim)[None, :] * stride_ckd
    tl.store(ck1_ptrs, y1, mask=mask_seq[:, None])
    tl.store(ck2_ptrs, y2, mask=mask_seq[:, None])

    # --- xv copy (no RoPE) ---
    xv_base = xv_ptr + batch_id * stride_xvb + head_id * stride_xvh
    cache_v_base = cache_v_ptr + batch_id * stride_cvb + head_id * stride_cvh

    xv1_ptrs = xv_base + seq_offs[:, None] * stride_xvs + cols[None, :] * stride_xvd
    cv1_ptrs = cache_v_base + abs_pos[:, None] * stride_cvs + cols[None, :] * stride_cvd
    xv1 = tl.load(xv1_ptrs, mask=mask_seq[:, None], other=0.0)
    tl.store(cv1_ptrs, xv1, mask=mask_seq[:, None])

    xv2_ptrs = xv_base + seq_offs[:, None] * stride_xvs + (cols + half_dim)[None, :] * stride_xvd
    cv2_ptrs = cache_v_base + abs_pos[:, None] * stride_cvs + (cols + half_dim)[None, :] * stride_cvd
    xv2 = tl.load(xv2_ptrs, mask=mask_seq[:, None], other=0.0)
    tl.store(cv2_ptrs, xv2, mask=mask_seq[:, None])


@triton.jit
def _rope_k_cache_write_scatter_kernel(
    xk_ptr, xv_ptr, cos_ptr, sin_ptr,
    cache_k_ptr, cache_v_ptr,
    scatter_positions_ptr,
    B, n_kv_heads, input_len, head_dim,
    stride_xkb, stride_xkh, stride_xks, stride_xkd,
    stride_xvb, stride_xvh, stride_xvs, stride_xvd,
    stride_ckb, stride_ckh, stride_cks, stride_ckd,
    stride_cvb, stride_cvh, stride_cvs, stride_cvd,
    stride_cs, stride_cd,
    BLOCK_SEQ: tl.constexpr,
    half_dim: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)

    batch_id = pid_bh // n_kv_heads
    head_id = pid_bh % n_kv_heads
    seq_start = pid_s * BLOCK_SEQ

    rows = tl.arange(0, BLOCK_SEQ)
    cols = tl.arange(0, half_dim)
    seq_offs = seq_start + rows

    mask_seq = seq_offs < input_len

    # scatter positions for each token in this tile
    abs_pos = tl.load(scatter_positions_ptr + seq_offs, mask=mask_seq, other=0)

    # --- xk RoPE ---
    xk_base = xk_ptr + batch_id * stride_xkb + head_id * stride_xkh
    x1_ptrs = xk_base + seq_offs[:, None] * stride_xks + cols[None, :] * stride_xkd
    x2_ptrs = xk_base + seq_offs[:, None] * stride_xks + (cols + half_dim)[None, :] * stride_xkd
    x1 = tl.load(x1_ptrs, mask=mask_seq[:, None], other=0.0)
    x2 = tl.load(x2_ptrs, mask=mask_seq[:, None], other=0.0)

    cos_ptrs = cos_ptr + seq_offs[:, None] * stride_cs + cols[None, :] * stride_cd
    sin_ptrs = sin_ptr + seq_offs[:, None] * stride_cs + cols[None, :] * stride_cd
    c = tl.load(cos_ptrs, mask=mask_seq[:, None], other=0.0)
    s = tl.load(sin_ptrs, mask=mask_seq[:, None], other=0.0)

    y1 = x1 * c - x2 * s
    y2 = x2 * c + x1 * s

    # write RoPE'd xk into cache_k at scatter positions
    cache_k_base = cache_k_ptr + batch_id * stride_ckb + head_id * stride_ckh
    ck1_ptrs = cache_k_base + abs_pos[:, None] * stride_cks + cols[None, :] * stride_ckd
    ck2_ptrs = cache_k_base + abs_pos[:, None] * stride_cks + (cols + half_dim)[None, :] * stride_ckd
    tl.store(ck1_ptrs, y1, mask=mask_seq[:, None])
    tl.store(ck2_ptrs, y2, mask=mask_seq[:, None])

    # --- xv copy (no RoPE) ---
    xv_base = xv_ptr + batch_id * stride_xvb + head_id * stride_xvh
    cache_v_base = cache_v_ptr + batch_id * stride_cvb + head_id * stride_cvh

    xv1_ptrs = xv_base + seq_offs[:, None] * stride_xvs + cols[None, :] * stride_xvd
    cv1_ptrs = cache_v_base + abs_pos[:, None] * stride_cvs + cols[None, :] * stride_cvd
    xv1 = tl.load(xv1_ptrs, mask=mask_seq[:, None], other=0.0)
    tl.store(cv1_ptrs, xv1, mask=mask_seq[:, None])

    xv2_ptrs = xv_base + seq_offs[:, None] * stride_xvs + (cols + half_dim)[None, :] * stride_xvd
    cv2_ptrs = cache_v_base + abs_pos[:, None] * stride_cvs + (cols + half_dim)[None, :] * stride_cvd
    xv2 = tl.load(xv2_ptrs, mask=mask_seq[:, None], other=0.0)
    tl.store(cv2_ptrs, xv2, mask=mask_seq[:, None])


def rope_and_cache_update(xk, xv, rope_cos, rope_sin, kv_cache, layer_idx, start_pos=None, scatter_positions=None):
    """
    Fused RoPE (for K) + KV-cache write.

    Applies RoPE to xk, then writes both (RoPE'd xk, xv) into the cache
    in a single kernel launch.  Dispatch variant based on argument type.

    Args:
        xk:                 [B, n_kv_heads, input_len, head_dim]
        xv:                 [B, n_kv_heads, input_len, head_dim]
        rope_cos:           [input_len, head_dim // 2]
        rope_sin:           [input_len, head_dim // 2]
        kv_cache:           KVCache instance
        layer_idx:          int
        start_pos:          int (sequential path)
        scatter_positions:  [input_len] 1-D tensor (CUDA Graph path)

    Returns:
        (cache_k, cache_v) — slice of cache visible to attention.
    """
    B, n_kv_heads, input_len, head_dim = xk.shape
    half_dim = head_dim // 2

    cache_k = kv_cache.k[layer_idx, :B]
    cache_v = kv_cache.v[layer_idx, :B]

    BLOCK_SEQ = 1 if input_len == 1 else 32
    grid = (B * n_kv_heads, (input_len + BLOCK_SEQ - 1) // BLOCK_SEQ)

    if scatter_positions is not None:
        _rope_k_cache_write_scatter_kernel[grid](
            xk, xv, rope_cos, rope_sin,
            cache_k, cache_v, scatter_positions,
            B, n_kv_heads, input_len, head_dim,
            xk.stride(0), xk.stride(1), xk.stride(2), xk.stride(3),
            xv.stride(0), xv.stride(1), xv.stride(2), xv.stride(3),
            cache_k.stride(0), cache_k.stride(1), cache_k.stride(2), cache_k.stride(3),
            cache_v.stride(0), cache_v.stride(1), cache_v.stride(2), cache_v.stride(3),
            rope_cos.stride(0), rope_cos.stride(1),
            BLOCK_SEQ=BLOCK_SEQ,
            half_dim=half_dim,
        )
        return cache_k, cache_v
    else:
        _rope_k_cache_write_seq_kernel[grid](
            xk, xv, rope_cos, rope_sin,
            cache_k, cache_v,
            B, n_kv_heads, input_len, head_dim,
            xk.stride(0), xk.stride(1), xk.stride(2), xk.stride(3),
            xv.stride(0), xv.stride(1), xv.stride(2), xv.stride(3),
            cache_k.stride(0), cache_k.stride(1), cache_k.stride(2), cache_k.stride(3),
            cache_v.stride(0), cache_v.stride(1), cache_v.stride(2), cache_v.stride(3),
            rope_cos.stride(0), rope_cos.stride(1),
            start_pos,
            BLOCK_SEQ=BLOCK_SEQ,
            half_dim=half_dim,
        )
        end_pos = start_pos + input_len
        return cache_k[:, :, :end_pos], cache_v[:, :, :end_pos]
