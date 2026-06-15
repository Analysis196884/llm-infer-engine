import torch
import torch.nn as nn

from .flash_attn import flash_attention
from .flash_decode import flash_decode
from .rms_norm import RMSNorm
from .rope import (
    apply_rotary,
    build_rope_cos_sin,
    precompute_freqs_cis,
    rope_and_cache_update,
)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        torch.cuda.nvtx.range_push("FFN")
        out = self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
        torch.cuda.nvtx.range_pop()
        return out
    
class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads  # Query heads
        self.n_kv_heads = args.n_kv_heads  # Key/Value heads
        self.head_dim = args.dim // args.n_heads
        
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(
        self,
        x,
        rope_cos,
        rope_sin,
        layer_idx=None,
        start_pos=0,
        kv_cache=None,
        use_cuda_graph=False,
        scatter_positions=None,
        decode_attn_mask=None,
    ):
        batch_size, query_len, _ = x.shape

        torch.cuda.nvtx.range_push("QKV_proj")
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        torch.cuda.nvtx.range_pop()

        xq = xq.view(batch_size, query_len, self.n_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(batch_size, query_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(batch_size, query_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        torch.cuda.nvtx.range_push("RoPE")
        xq = apply_rotary(xq, rope_cos, rope_sin)
        torch.cuda.nvtx.range_pop()

        if kv_cache is not None:
            torch.cuda.nvtx.range_push("KV_cache")
            if layer_idx is None:
                raise ValueError("layer_idx is required when using kv_cache")
            xk, xv = rope_and_cache_update(
                xk, xv, rope_cos, rope_sin,
                kv_cache, layer_idx,
                start_pos=start_pos,
                scatter_positions=scatter_positions if use_cuda_graph else None,
            )
            torch.cuda.nvtx.range_pop()

        if query_len > 1:
            torch.cuda.nvtx.range_push("FlashAttn")
            output = flash_attention(xq, xk, xv)
            torch.cuda.nvtx.range_pop()
        else:
            torch.cuda.nvtx.range_push("FlashDecode")
            output = flash_decode(xq, xk, xv, mask=decode_attn_mask)
            torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("Output_proj")
        output = output.transpose(1, 2).contiguous().view(batch_size, query_len, -1)
        out = self.wo(output)
        torch.cuda.nvtx.range_pop()
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args.dim, args.hidden_dim)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x,
        rope_cos,
        rope_sin,
        layer_idx=None,
        start_pos=0,
        kv_cache=None,
        use_cuda_graph=False,
        scatter_positions=None,
        decode_attn_mask=None,
    ):
        torch.cuda.nvtx.range_push(f"Attn_{layer_idx}")
        x = x + self.attention(
            self.attention_norm(x),
            rope_cos,
            rope_sin,
            layer_idx=layer_idx,
            start_pos=start_pos,
            kv_cache=kv_cache,
            use_cuda_graph=use_cuda_graph,
            scatter_positions=scatter_positions,
            decode_attn_mask=decode_attn_mask,
        )
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push(f"FFN_{layer_idx}")
        x = x + self.feed_forward(self.ffn_norm(x))
        torch.cuda.nvtx.range_pop()
        return x
    
class Llama3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(
            args.dim // args.n_heads,
            args.max_seq_len,
            theta=args.rope_theta,
            rope_scaling=args.rope_scaling,
        )

    def forward(
        self,
        tokens,
        start_pos=0,
        kv_cache=None,
        freqs_cis=None,
        use_cuda_graph=False,
        scatter_positions=None,
        decode_attn_mask=None,
    ):
        torch.cuda.nvtx.range_push("Embed")
        h = self.tok_embeddings(tokens)
        torch.cuda.nvtx.range_pop()

        seq_len = tokens.shape[1]
        if freqs_cis is None:
            freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]
        rope_cos, rope_sin = build_rope_cos_sin(freqs_cis, dtype=h.dtype, device=h.device)

        torch.cuda.nvtx.range_push("Layers")
        for layer_idx, layer in enumerate(self.layers):
            h = layer(
                h,
                rope_cos,
                rope_sin,
                layer_idx=layer_idx,
                start_pos=start_pos,
                kv_cache=kv_cache,
                use_cuda_graph=use_cuda_graph,
                scatter_positions=scatter_positions,
                decode_attn_mask=decode_attn_mask,
            )
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("Output")
        if seq_len > 1:
            h = h[:, -1:, :]
        out = self.output(self.norm(h))
        torch.cuda.nvtx.range_pop()
        return out
