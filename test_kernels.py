import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from src.config import ModelArgs
from src.kv_cache import KVCache
from src.loader import _to_hf_name, load_weights
from src.model import Llama3, RMSNorm, apply_rotary_emb, precompute_freqs_cis
from src.sampler import sample
from main import _normalize_input_ids, _parse_env_file, _resolve_runtime_args


class TestModelKernels(unittest.TestCase):
	def test_rmsnorm_shape_and_finite(self):
		norm = RMSNorm(dim=16)
		x = torch.randn(2, 3, 16)
		y = norm(x)
		self.assertEqual(y.shape, x.shape)
		self.assertTrue(torch.isfinite(y).all().item())

	def test_rotary_emb_shape(self):
		batch_size, seq_len, n_heads, head_dim = 1, 5, 4, 8
		xq = torch.randn(batch_size, seq_len, n_heads, head_dim)
		xk = torch.randn(batch_size, seq_len, n_heads, head_dim)
		freqs_cis = precompute_freqs_cis(head_dim, seq_len)
		yq, yk = apply_rotary_emb(xq, xk, freqs_cis)
		self.assertEqual(yq.shape, xq.shape)
		self.assertEqual(yk.shape, xk.shape)

	def test_sampler_greedy(self):
		logits = torch.tensor([[[0.1, 1.0, 0.5]]], dtype=torch.float32)
		token = sample(logits, temperature=0.0)
		self.assertEqual(token.item(), 1)

	def test_sampler_top_p_batch_safe(self):
		torch.manual_seed(0)
		logits = torch.tensor(
			[
				[[2.0, 1.0, 0.2, -1.0]],
				[[0.1, 3.0, 0.3, 0.0]],
			],
			dtype=torch.float32,
		)
		tokens = sample(logits, temperature=0.8, top_p=0.9)
		self.assertEqual(tokens.shape, (2, 1))
		self.assertTrue(torch.all(tokens >= 0).item())
		self.assertTrue(torch.all(tokens < logits.size(-1)).item())

	def test_sampler_top_p_edge_small_threshold(self):
		logits = torch.tensor([[[1.2, 0.7, 0.3]]], dtype=torch.float32)
		tokens = sample(logits, temperature=0.7, top_p=1e-6)
		self.assertEqual(tokens.shape, (1, 1))
		self.assertTrue(0 <= int(tokens.item()) < 3)

	def test_hf_name_mapping(self):
		self.assertEqual(_to_hf_name("tok_embeddings.weight"), "model.embed_tokens.weight")
		self.assertEqual(
			_to_hf_name("layers.0.attention.wq.weight"),
			"model.layers.0.self_attn.q_proj.weight",
		)

	def test_load_weights_roundtrip(self):
		args = ModelArgs(
			dim=32,
			n_layers=1,
			n_heads=4,
			n_kv_heads=4,
			hidden_dim=64,
			vocab_size=128,
			max_seq_len=64,
			device="cpu",
		)
		model = Llama3(args)

		state_dict = {k: torch.randn_like(v) for k, v in model.state_dict().items()}

		with tempfile.TemporaryDirectory() as tmpdir:
			weights_path = str(Path(tmpdir) / "toy.safetensors")
			save_file(state_dict, weights_path)
			report = load_weights(model, weights_path, device="cpu")

		self.assertEqual(report["loaded"], len(state_dict))
		self.assertEqual(report["missing"], [])

	def test_kv_cache_incremental_matches_full(self):
		args = ModelArgs(
			dim=32,
			n_layers=2,
			n_heads=4,
			n_kv_heads=4,
			hidden_dim=64,
			vocab_size=128,
			max_seq_len=64,
			device="cpu",
		)
		model = Llama3(args).eval()
		tokens = torch.randint(0, args.vocab_size, (1, 6), dtype=torch.long)

		with torch.no_grad():
			full_logits = model(tokens)

			cache = KVCache(args)
			prefill_logits = model(tokens[:, :4], start_pos=0, kv_cache=cache)
			step1_logits = model(tokens[:, 4:5], start_pos=4, kv_cache=cache)
			step2_logits = model(tokens[:, 5:6], start_pos=5, kv_cache=cache)

			merged_logits = torch.cat([prefill_logits, step1_logits, step2_logits], dim=1)

		self.assertTrue(torch.allclose(full_logits, merged_logits, atol=1e-5, rtol=1e-5))

	def test_parse_env_file(self):
		with tempfile.TemporaryDirectory() as tmpdir:
			env_path = Path(tmpdir) / ".env"
			env_path.write_text(
				"\n".join(
					[
						"# comment",
						"LLM_TOKENIZER=meta-llama/Llama-3.2-1B",
						"export LLM_PROMPT='hello world'",
						"LLM_MAX_NEW_TOKENS=16",
					]
				),
				encoding="utf-8",
			)

			env_data = _parse_env_file(str(env_path))

		self.assertEqual(env_data["LLM_TOKENIZER"], "meta-llama/Llama-3.2-1B")
		self.assertEqual(env_data["LLM_PROMPT"], "hello world")
		self.assertEqual(env_data["LLM_MAX_NEW_TOKENS"], "16")

	def test_resolve_runtime_args_from_env(self):
		args = SimpleNamespace(
			tokenizer=None,
			weights=None,
			prompt=None,
			system_prompt=None,
			max_new_tokens=None,
			temperature=None,
			top_p=None,
			device=None,
			seed=None,
			dim=None,
			n_layers=None,
			n_heads=None,
			n_kv_heads=None,
			hidden_dim=None,
			vocab_size=None,
			max_seq_len=None,
		)
		env_data = {
			"LLM_TOKENIZER": "meta-llama/Llama-3.2-1B",
			"LLM_PROMPT": "from env",
			"LLM_SYSTEM_PROMPT": "sys from env",
			"LLM_MAX_NEW_TOKENS": "32",
			"LLM_TEMPERATURE": "0.6",
		}

		resolved = _resolve_runtime_args(args, env_data)

		self.assertEqual(resolved["tokenizer"], "meta-llama/Llama-3.2-1B")
		self.assertEqual(resolved["prompt"], "from env")
		self.assertEqual(resolved["system_prompt"], "sys from env")
		self.assertEqual(resolved["max_new_tokens"], 32)
		self.assertAlmostEqual(resolved["temperature"], 0.6)

	def test_resolve_runtime_args_cli_override(self):
		args = SimpleNamespace(
			tokenizer="cli-tokenizer",
			weights=None,
			prompt="cli prompt",
			system_prompt="cli system",
			max_new_tokens=8,
			temperature=None,
			top_p=None,
			device="cpu",
			seed=None,
			dim=None,
			n_layers=None,
			n_heads=None,
			n_kv_heads=None,
			hidden_dim=None,
			vocab_size=None,
			max_seq_len=None,
		)
		env_data = {
			"LLM_TOKENIZER": "env-tokenizer",
			"LLM_PROMPT": "env prompt",
			"LLM_SYSTEM_PROMPT": "env system",
			"LLM_MAX_NEW_TOKENS": "64",
			"LLM_DEVICE": "cuda",
		}

		resolved = _resolve_runtime_args(args, env_data)

		self.assertEqual(resolved["tokenizer"], "cli-tokenizer")
		self.assertEqual(resolved["prompt"], "cli prompt")
		self.assertEqual(resolved["system_prompt"], "cli system")
		self.assertEqual(resolved["max_new_tokens"], 8)
		self.assertEqual(resolved["device"], "cpu")

	def test_normalize_input_ids_tensor(self):
		encoded = torch.tensor([[1, 2, 3]], dtype=torch.long)
		self.assertEqual(_normalize_input_ids(encoded), [1, 2, 3])

	def test_normalize_input_ids_nested_list(self):
		encoded = [[4, 5, 6]]
		self.assertEqual(_normalize_input_ids(encoded), [4, 5, 6])


if __name__ == "__main__":
	unittest.main()
