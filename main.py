import argparse
import json
import os
from pathlib import Path

import torch

from src.config import ModelArgs
from src.loader import load_weights
from src.model import Llama3
from src.sampler import sample
from src.tokenizer import Tokenizer


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Tiny Llama inference entrypoint")
	parser.add_argument("--env-file", type=str, default=".env", help="Path to .env file, default: .env")
	parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path or HF model id")
	parser.add_argument("--weights", type=str, default=None, help="Path to safetensors weights")
	parser.add_argument("--prompt", type=str, default=None, help="Input prompt")
	parser.add_argument("--system-prompt", type=str, default=None, help="System prompt for chat template")
	parser.add_argument("--max-new-tokens", type=int, default=None, help="Maximum generated tokens")
	parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
	parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling threshold")
	parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cpu/cuda")
	parser.add_argument("--seed", type=int, default=None, help="Random seed")
	parser.add_argument("--dim", type=int, default=None)
	parser.add_argument("--n-layers", type=int, default=None)
	parser.add_argument("--n-heads", type=int, default=None)
	parser.add_argument("--n-kv-heads", type=int, default=None)
	parser.add_argument("--hidden-dim", type=int, default=None)
	parser.add_argument("--vocab-size", type=int, default=None)
	parser.add_argument("--max-seq-len", type=int, default=None)
	return parser


def _parse_env_file(env_file: str) -> dict:
	if not env_file:
		return {}
	path = Path(env_file)
	if not path.exists() or not path.is_file():
		return {}

	env_data = {}
	for raw_line in path.read_text(encoding="utf-8").splitlines():
		line = raw_line.strip()
		if not line or line.startswith("#"):
			continue
		if line.startswith("export "):
			line = line[len("export ") :].strip()
		if "=" not in line:
			continue

		key, value = line.split("=", 1)
		key = key.strip()
		value = value.strip()
		if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
			value = value[1:-1]
		env_data[key] = value

	return env_data


def _resolve_option(cli_value, env_key: str, env_file_data: dict, default, cast):
	if cli_value is not None:
		return cli_value

	raw = env_file_data.get(env_key)
	if raw is None:
		raw = os.environ.get(env_key)

	if raw is None or raw == "":
		return default

	if cast is str:
		return raw

	return cast(raw)


def _resolve_runtime_args(args, env_file_data: dict) -> dict:
	defaults = {
		"weights": "",
		"system_prompt": "You are a helpful assistant.",
		"max_new_tokens": 64,
		"temperature": 0.0,
		"top_p": 1.0,
		"seed": 42,
		"dim": 2048,
		"n_layers": 16,
		"n_heads": 32,
		"n_kv_heads": 8,
		"hidden_dim": 8192,
		"vocab_size": 128256,
		"max_seq_len": 4096,
	}

	resolved = {
		"tokenizer": _resolve_option(args.tokenizer, "LLM_TOKENIZER", env_file_data, None, str),
		"weights": _resolve_option(args.weights, "LLM_WEIGHTS", env_file_data, defaults["weights"], str),
		"prompt": _resolve_option(args.prompt, "LLM_PROMPT", env_file_data, None, str),
		"system_prompt": _resolve_option(args.system_prompt, "LLM_SYSTEM_PROMPT", env_file_data, defaults["system_prompt"], str),
		"max_new_tokens": _resolve_option(args.max_new_tokens, "LLM_MAX_NEW_TOKENS", env_file_data, defaults["max_new_tokens"], int),
		"temperature": _resolve_option(args.temperature, "LLM_TEMPERATURE", env_file_data, defaults["temperature"], float),
		"top_p": _resolve_option(args.top_p, "LLM_TOP_P", env_file_data, defaults["top_p"], float),
		"device": _resolve_option(args.device, "LLM_DEVICE", env_file_data, None, str),
		"seed": _resolve_option(args.seed, "LLM_SEED", env_file_data, defaults["seed"], int),
		"dim": _resolve_option(args.dim, "LLM_DIM", env_file_data, defaults["dim"], int),
		"n_layers": _resolve_option(args.n_layers, "LLM_N_LAYERS", env_file_data, defaults["n_layers"], int),
		"n_heads": _resolve_option(args.n_heads, "LLM_N_HEADS", env_file_data, defaults["n_heads"], int),
		"n_kv_heads": _resolve_option(args.n_kv_heads, "LLM_N_KV_HEADS", env_file_data, defaults["n_kv_heads"], int),
		"hidden_dim": _resolve_option(args.hidden_dim, "LLM_HIDDEN_DIM", env_file_data, defaults["hidden_dim"], int),
		"vocab_size": _resolve_option(args.vocab_size, "LLM_VOCAB_SIZE", env_file_data, defaults["vocab_size"], int),
		"max_seq_len": _resolve_option(args.max_seq_len, "LLM_MAX_SEQ_LEN", env_file_data, defaults["max_seq_len"], int),
	}

	return resolved


def _load_hf_model_config(model_path: str) -> dict:
	config_path = Path(model_path) / "config.json"
	if not config_path.exists() or not config_path.is_file():
		return {}
	try:
		return json.loads(config_path.read_text(encoding="utf-8"))
	except Exception:
		return {}

def _normalize_input_ids(encoded_ids):
	if isinstance(encoded_ids, torch.Tensor):
		if encoded_ids.dim() == 2:
			encoded_ids = encoded_ids[0]
		return [int(item) for item in encoded_ids.tolist()]

	if isinstance(encoded_ids, dict):
		if "input_ids" not in encoded_ids:
			raise ValueError("Tokenizer output dict does not contain input_ids")
		return _normalize_input_ids(encoded_ids["input_ids"])

	if hasattr(encoded_ids, "keys") and "input_ids" in encoded_ids:
		return _normalize_input_ids(encoded_ids["input_ids"])

	if isinstance(encoded_ids, str):
		raise ValueError("Tokenizer returned string instead of token ids")

	if isinstance(encoded_ids, list):
		if len(encoded_ids) == 0:
			return []
		if isinstance(encoded_ids[0], list):
			return [int(item) for item in encoded_ids[0]]
		return [int(item) for item in encoded_ids]

	raise TypeError(f"Unsupported tokenizer output type: {type(encoded_ids)}")


def build_chat_input_ids(tokenizer: Tokenizer, user_prompt: str, system_prompt: str):
	inner_tokenizer = tokenizer.tokenizer
	if not hasattr(inner_tokenizer, "apply_chat_template"):
		raise RuntimeError("Current tokenizer does not support apply_chat_template")

	messages = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_prompt},
	]
	encoded = inner_tokenizer.apply_chat_template(
		messages,
		add_generation_prompt=True,
		return_tensors="pt",
	)
	return _normalize_input_ids(encoded)


def generate_text(model, tokenizer: Tokenizer, user_prompt: str, system_prompt: str, max_new_tokens: int, temperature: float, top_p: float, device: str):
	input_ids = build_chat_input_ids(tokenizer, user_prompt=user_prompt, system_prompt=system_prompt)
	if len(input_ids) == 0:
		raise ValueError("Prompt is empty after tokenization.")

	generated = list(input_ids)
	prompt_len = len(input_ids)

	with torch.no_grad():
		for _ in range(max_new_tokens):
			tokens = torch.tensor([generated], dtype=torch.long, device=device)
			logits = model(tokens)
			next_token = sample(logits, temperature=temperature, top_k=top_p)
			next_id = int(next_token.item())
			generated.append(next_id)

			if tokenizer.eos_id is not None and next_id == tokenizer.eos_id:
				break

			if len(generated) >= model.freqs_cis.size(0):
				break

	new_token_ids = generated[prompt_len:]
	return tokenizer.decode(new_token_ids)


def main():
	parser = _build_parser()
	args = parser.parse_args()
	env_file_data = _parse_env_file(args.env_file)
	resolved = _resolve_runtime_args(args, env_file_data)

	if not resolved["tokenizer"]:
		parser.error("Missing tokenizer. Provide --tokenizer or set LLM_TOKENIZER in .env/environment.")
	if not resolved["prompt"]:
		parser.error("Missing prompt. Provide --prompt or set LLM_PROMPT in .env/environment.")

	torch.manual_seed(resolved["seed"])

	if resolved["device"]:
		device = resolved["device"]
	else:
		device = "cuda" if torch.cuda.is_available() else "cpu"

	model_args = ModelArgs(
		dim=resolved["dim"],
		n_layers=resolved["n_layers"],
		n_heads=resolved["n_heads"],
		n_kv_heads=resolved["n_kv_heads"],
		vocab_size=resolved["vocab_size"],
		hidden_dim=resolved["hidden_dim"],
		max_seq_len=resolved["max_seq_len"],
		rope_theta=500000.0,
		rope_scaling=None,
		device=device,
	)

	hf_cfg = _load_hf_model_config(resolved["tokenizer"])
	if hf_cfg:
		model_args.norm_eps = hf_cfg.get("rms_norm_eps", model_args.norm_eps)
		model_args.rope_theta = hf_cfg.get("rope_theta", model_args.rope_theta)
		model_args.rope_scaling = hf_cfg.get("rope_scaling", model_args.rope_scaling)
		model_args.max_seq_len = min(
			model_args.max_seq_len,
			hf_cfg.get("max_position_embeddings", model_args.max_seq_len),
		)

	model = Llama3(model_args).to(device)
	tokenizer = Tokenizer(resolved["tokenizer"])

	if resolved["weights"]:
		report = load_weights(model, resolved["weights"], device)
		print(f"[loader] loaded={report['loaded']} missing={len(report['missing'])} unexpected={len(report['unexpected'])}")
	else:
		print("[loader] no weights provided, using randomly initialized model")

	print("System Prompt:", resolved["system_prompt"])
	print("User Prompt:", resolved["prompt"])

	text = generate_text(
		model=model,
		tokenizer=tokenizer,
		user_prompt=resolved["prompt"],
		system_prompt=resolved["system_prompt"],
		max_new_tokens=resolved["max_new_tokens"],
		temperature=resolved["temperature"],
		top_p=resolved["top_p"],
		device=device,
	)
	print(text)


if __name__ == "__main__":
	main()
