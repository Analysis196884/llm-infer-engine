import torch
import torch.nn as nn

from src.config import ModelArgs
from src.kv_cache import KVCache


class CUDAGraphDecodeRunner:
    def __init__(self, model: nn.Module, model_args: ModelArgs, kv_cache: KVCache, device: str):
        if not device.startswith("cuda"):
            raise ValueError("CUDAGraphDecodeRunner requires CUDA device")

        self.model = model
        self.model_args = model_args
        self.kv_cache = kv_cache
        self.device = device
        self.graph = torch.cuda.CUDAGraph()
        head_dim = model_args.dim // model_args.n_heads
        self.static_input_token = torch.zeros((1, 1), dtype=torch.long, device=device)
        self.static_scatter_positions = torch.zeros((1,), dtype=torch.long, device=device)
        self.static_freqs_cis = torch.zeros((1, head_dim // 2), dtype=model.freqs_cis.dtype, device=device)
        self.static_decode_mask = torch.zeros((1, 1, 1, model_args.max_seq_len), dtype=torch.bool, device=device)
        self.static_logits = None

        self._capture()

    def _capture(self):
        self.static_decode_mask[..., 0] = True
        self.static_freqs_cis.copy_(self.model.freqs_cis[:1].to(self.device))

        torch.cuda.nvtx.range_push("CUDAGraphWarmup")
        warmup_stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(warmup_stream):
            _ = self.model(
                self.static_input_token,
                kv_cache=self.kv_cache,
                freqs_cis=self.static_freqs_cis,
                use_cuda_graph=True,
                scatter_positions=self.static_scatter_positions,
                decode_attn_mask=self.static_decode_mask,
            )
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("CUDAGraphCapture")
        with torch.cuda.graph(self.graph):
            self.static_logits = self.model(
                self.static_input_token,
                kv_cache=self.kv_cache,
                freqs_cis=self.static_freqs_cis,
                use_cuda_graph=True,
                scatter_positions=self.static_scatter_positions,
                decode_attn_mask=self.static_decode_mask,
            )
        torch.cuda.nvtx.range_pop()

    def decode(self, token_id: int, current_pos: int) -> torch.Tensor:
        if current_pos >= self.model_args.max_seq_len:
            raise ValueError(f"current_pos out of range: {current_pos}")

        torch.cuda.nvtx.range_push("CUDAGraphReplay")
        self.static_input_token[0, 0] = token_id
        self.static_scatter_positions[0] = current_pos
        self.static_freqs_cis.copy_(self.model.freqs_cis[current_pos : current_pos + 1].to(self.device))
        self.static_decode_mask.zero_()
        self.static_decode_mask[..., : current_pos + 1] = True

        self.graph.replay()
        torch.cuda.nvtx.range_pop()
        return self.static_logits
