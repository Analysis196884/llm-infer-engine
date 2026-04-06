from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 2048          
    n_layers: int = 16       
    n_heads: int = 32        
    n_kv_heads: int = 8     
    vocab_size: int = 128256 
    hidden_dim: int = 8192
    norm_eps: float = 1e-5   
    max_batch_size: int = 1
    max_seq_len: int = 4096 
    rope_theta: float = 500000.0
    rope_scaling: Optional[dict] = None
    device: str = "cuda"