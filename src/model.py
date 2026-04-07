import torch
import torch.nn as nn
import math

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Compute RMS in fp32 for numerical stability, then cast back
        x_float = x.float()
        norm_x = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm_x * self.weight.float()).to(dtype=x.dtype)
    
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
    # Convert to complex numbers in favor of rotation operations
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(xq, xk, freqs_cis):
    cos_half = freqs_cis.real
    sin_half = freqs_cis.imag

    cos = torch.cat([cos_half, cos_half], dim=-1).unsqueeze(0).unsqueeze(2)
    sin = torch.cat([sin_half, sin_half], dim=-1).unsqueeze(0).unsqueeze(2)

    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        # SwiGLU structure: (Swish(W1x) * W3x) * W2
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
    
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

    def forward(self, x, freqs_cis, layer_idx=None, start_pos=0, kv_cache=None):
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # apply rotary embeddings to Q and K
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        if kv_cache is not None:
            if layer_idx is None:
                raise ValueError("layer_idx is required when using kv_cache")
            xk, xv = kv_cache.update(layer_idx, start_pos, xk, xv)

        # GQA process: repeat K and V for each Q head
        if self.n_kv_heads != self.n_heads:
            xk = xk.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)
            xv = xv.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)
        
        # Compute Scaled Dot-Product Attention
        q = xq.transpose(1, 2).float()
        k = xk.transpose(1, 2).float()
        v = xv.transpose(1, 2).float()

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        total_len = xk.size(1)
        query_pos = torch.arange(start_pos, start_pos + seq_len, device=x.device).unsqueeze(-1)
        key_pos = torch.arange(total_len, device=x.device).unsqueeze(0)
        causal_mask = key_pos > query_pos
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        scores = torch.softmax(scores, dim=-1)
        output = torch.matmul(scores, v).to(dtype=xq.dtype)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)
    
class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args.dim, args.hidden_dim)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cis, layer_idx=None, start_pos=0, kv_cache=None):
        # residual connection: x = x + Attn(Norm(x))
        x = x + self.attention(
            self.attention_norm(x),
            freqs_cis,
            layer_idx=layer_idx,
            start_pos=start_pos,
            kv_cache=kv_cache,
        )
        # residual connection: x = x + FFN(Norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
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

    def forward(self, tokens, start_pos=0, kv_cache=None):
        h = self.tok_embeddings(tokens)
        seq_len = tokens.shape[1]
        freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len].to(h.device)
        
        for layer_idx, layer in enumerate(self.layers):
            h = layer(h, freqs_cis, layer_idx=layer_idx, start_pos=start_pos, kv_cache=kv_cache)
            
        return self.output(self.norm(h))