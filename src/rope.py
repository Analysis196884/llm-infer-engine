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
