import torch
import triton
import triton.language as tl

@triton.jit
def flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    batch_size, seq_len, n_heads, n_kv_heads, head_dim: tl.constexpr,
    stride_qb, stride_ql, stride_qh, stride_qd,
    stride_kb, stride_kl, stride_kh, stride_kd,
    stride_vb, stride_vl, stride_vh, stride_vd,
    stride_ob, stride_ol, stride_oh, stride_od,
    BLOCK_SIZE: tl.constexpr
):
    pid_bh = tl.program_id(0)
    batch_id = pid_bh // n_heads
    head_id = pid_bh % n_heads
    row_q_start = tl.program_id(1) * BLOCK_SIZE
    row_k_start = 0

    q_offset = batch_id * stride_qb + head_id * stride_qh + row_q_start * stride_ql
    k_offset = batch_id * stride_kb + head_id * stride_kh
    v_offset = batch_id * stride_vb + head_id * stride_vh
    out_offset = batch_id * stride_ob + head_id * stride_oh + row_q_start * stride_ol

    rows_q = tl.arange(0, BLOCK_SIZE)
    rows_k = tl.arange(0, BLOCK_SIZE)

    q_ptrs = q_ptr + q_offset + rows_q[:, None] * stride_ql + tl.arange(0, head_dim)[None, :] * stride_qd
    k_ptrs = k_ptr + k_offset + rows_k[:, None] * stride_kl + tl.arange(0, head_dim)[None, :] * stride_kd
    v_ptrs = v_ptr + v_offset + rows_k[:, None] * stride_vl + tl.arange(0, head_dim)[None, :] * stride_vd
    out_ptrs = out_ptr + out_offset + rows_q[:, None] * stride_ol + tl.arange(0, head_dim)[None, :] * stride_od

    # Initialize intermediate variables
    m = tl.full([BLOCK_SIZE], float("-inf"), dtype=tl.float32) # max score for each row
    l = tl.zeros([BLOCK_SIZE], dtype=tl.float32)               # sum of exp(scores) for each row 
    acc = tl.zeros([BLOCK_SIZE, head_dim], dtype=tl.float32)   # unnormalized output accumulator

    # load Q for current block of rows
    q = tl.load(q_ptrs, mask=(row_q_start + rows_q[:, None]) < seq_len, other=0.0)

    # loop over blocks of K and V along sequence dimension
    for start_col in range(0, seq_len, BLOCK_SIZE):
        k = tl.load(k_ptrs, mask=(start_col + rows_k[:, None]) < seq_len, other=0.0)
        v = tl.load(v_ptrs, mask=(start_col + rows_k[:, None]) < seq_len, other=0.0)

        # causal mask: mask out upper triangular part of attention scores
        i = row_q_start + tl.arange(0, BLOCK_SIZE)
        j = start_col + tl.arange(0, BLOCK_SIZE)
        mask = j[None, :] > i[:, None]

        # attention scores: s = q @ k.T / sqrt(head_dim)
        s = tl.dot(q, tl.trans(k)) / (head_dim ** 0.5) # BLOCK x BLOCK
        s = tl.where(mask, -1.0e9, s)
        # also mask out of bounds K
        s = tl.where((start_col + rows_k[None, :]) >= seq_len, -1.0e9, s)

        # intermediate variables for current block
        m_block = tl.max(s, axis=1)
        # handle max being -1.0e9 (all masked)
        m_block = tl.maximum(m_block, -1.0e9)
        
        p_block = tl.exp(s - m_block[:, None])
        l_block = tl.sum(p_block, axis=1)

        # update m, l and out
        m_new = tl.maximum(m, m_block)
        
        alpha = tl.exp(m - m_new)
        beta = tl.exp(m_block - m_new)
        
        l_new = alpha * l + beta * l_block
        # Ensure p_block and v have same dtype for tl.dot. v is loaded from K/V cache which might be fp16.
        acc = acc * alpha[:, None] + tl.dot(p_block.to(v.dtype), v) * beta[:, None]
        
        m = m_new
        l = l_new

        # update ptrs
        k_ptrs += BLOCK_SIZE * stride_kl
        v_ptrs += BLOCK_SIZE * stride_vl

        row_k_start += BLOCK_SIZE

    # write output
    out = acc / l[:, None]
    tl.store(out_ptrs, out, mask=(row_q_start + rows_q[:, None]) < seq_len)

def flash_attention(q, k, v):
    # expect q: [batch, seq_len, n_heads, head_dim]
    # expect k,v: [batch, seq_len, n_kv_heads, head_dim]
    batch_size, seq_len, n_heads, head_dim = q.shape
    _, kv_seq_len, n_kv_heads, _ = k.shape

    # handle GQA/MQA by repeating K/V if necessary
    if n_heads != n_kv_heads:
        k = k[:, :, :, None, :].expand(batch_size, kv_seq_len, n_kv_heads, n_heads // n_kv_heads, head_dim).reshape(batch_size, kv_seq_len, n_heads, head_dim)
        v = v[:, :, :, None, :].expand(batch_size, kv_seq_len, n_kv_heads, n_heads // n_kv_heads, head_dim).reshape(batch_size, kv_seq_len, n_heads, head_dim)

    out = torch.empty_like(q)

    grid = (
        batch_size * n_heads,
        (seq_len + 63) // 64,
    )
    flash_attention_kernel[grid](
        q, k, v, out,
        batch_size, seq_len, n_heads, n_kv_heads, head_dim,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_SIZE=64
    )
    return out

def test_flash_attention():
    batch_size = 2
    seq_len = 128
    n_heads = 4
    head_dim = 64

    q = torch.randn(batch_size, seq_len, n_heads, head_dim, device="cuda")
    k = torch.randn(batch_size, seq_len, n_heads, head_dim, device="cuda")
    v = torch.randn(batch_size, seq_len, n_heads, head_dim, device="cuda")

    # q = torch.ones(batch_size, seq_len, n_heads, head_dim, device="cuda")
    # k = torch.ones(batch_size, seq_len, n_heads, head_dim, device="cuda")
    # v = torch.ones(batch_size, seq_len, n_heads, head_dim, device="cuda")

    out_flash = flash_attention(q, k, v)
    out_ref = torch.nn.functional.scaled_dot_product_attention(q.view(batch_size * n_heads, seq_len, head_dim), 
                                                              k.view(batch_size * n_heads, seq_len, head_dim), 
                                                              v.view(batch_size * n_heads, seq_len, head_dim), 
                                                              attn_mask=torch.triu(torch.ones(seq_len, seq_len, device=q.device) * float("-inf"), diagonal=1))
    out_ref = out_ref.view(batch_size, n_heads, seq_len, head_dim).transpose(1, 2)

    # print the results for debugging
    print("Output from flash attention:\n", out_flash)
    print("Output from reference implementation:\n", out_ref)

    assert torch.allclose(out_flash.cpu(), out_ref.cpu(), atol=1e-3, rtol=1e-3), "Flash attention does not match reference implementation"


if __name__ == "__main__":
    test_flash_attention()
    print("Flash attention test passed!")