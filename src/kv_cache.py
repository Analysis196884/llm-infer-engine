import torch
from src.config import ModelArgs

class KVCache:
    def __init__(self, args: ModelArgs):
        self.n_layers = args.n_layers
        self.max_batch_size = args.max_batch_size
        self.max_seq_len = args.max_seq_len
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        # pre-allocate KV cache for maximum sequence length
        shape = (self.n_layers, self.max_batch_size, self.max_seq_len, self.n_kv_heads, self.head_dim)
        self.k = torch.zeros(shape, device=args.device)
        self.v = torch.zeros(shape, device=args.device)

    def reset(self):
        self.k.zero_()
        self.v.zero_()

    def update(self, layer_idx, start_pos, k_val, v_val):
        # k_val/v_val shape: (batch, seq_len, n_kv_heads, head_dim)
        batch_size = k_val.size(0)
        seq_len = k_val.size(1)
        end_pos = start_pos + seq_len

        self.k[layer_idx, :batch_size, start_pos:end_pos] = k_val
        self.v[layer_idx, :batch_size, start_pos:end_pos] = v_val

        return (
            self.k[layer_idx, :batch_size, :end_pos],
            self.v[layer_idx, :batch_size, :end_pos],
        )