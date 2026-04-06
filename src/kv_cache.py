import torch
from config import ModelArgs

class KVCache:
    def __init__(self, args: ModelArgs):
        self.max_seq_len = args.max_seq_len
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.dim // args.n_kv_heads

        # pre-allocate KV cache for maximum sequence length
        shape = (args.max_batch_size, self.max_seq_len, self.n_kv_heads, self.head_dim)
        self.k = torch.zeros(shape).to(args.device)
        self.v = torch.zeros(shape).to(args.device)

    def update(self, layer_idx, pos, k_val, v_val):
        # k_val/v_val shape: (batch, seq_len, n_kv_heads, head_dim)
        seq_len = k_val.size(1)
        self.k[layer_idx, :, pos : pos + seq_len] = k_val
        self.v[layer_idx, :, pos : pos + seq_len] = v_val

        return self.k[layer_idx, :, : pos + seq_len], self.v[layer_idx, :, : pos + seq_len]