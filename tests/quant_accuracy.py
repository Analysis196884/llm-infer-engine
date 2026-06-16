import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ModelArgs
from src.loader import build_model_from_weights
from src.model import Llama3
from src.model_quantized import Llama3Quantized
from src.quant_utils import (
    cosine_similarity,
    logits_max_abs_error,
    logits_relative_error,
    perplexity,
    top1_agreement,
)
from src.sampler import sample
from src.tokenizer import Tokenizer


def _load_env(env_file: str) -> dict:
    path = Path(env_file)
    if not path.exists():
        return {}
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        env[key] = value
    return env


def _resolve_option(cli_value, env_key: str, env: dict, default, cast):
    if cli_value is not None:
        return cast(cli_value)
    raw = env.get(env_key)
    if raw is None or raw == "":
        return default
    return cast(raw)


def _load_model(weights_path: str, tokenizer_path: str, device: str, quantization: Optional[str], max_seq_len: int):
    cfg_path = Path(tokenizer_path) / "config.json"
    hf_cfg = {}
    if cfg_path.exists():
        hf_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    model_args = ModelArgs(
        dim=hf_cfg.get("hidden_size", ModelArgs.dim),
        n_layers=hf_cfg.get("num_hidden_layers", ModelArgs.n_layers),
        n_heads=hf_cfg.get("num_attention_heads", ModelArgs.n_heads),
        n_kv_heads=hf_cfg.get("num_key_value_heads", ModelArgs.n_kv_heads),
        vocab_size=hf_cfg.get("vocab_size", ModelArgs.vocab_size),
        hidden_dim=hf_cfg.get("intermediate_size", ModelArgs.hidden_dim),
        max_seq_len=max_seq_len,
        rope_theta=hf_cfg.get("rope_theta", 500000.0),
        rope_scaling=hf_cfg.get("rope_scaling", None),
        device=device,
        quantization=quantization,
    )
    model_args.norm_eps = hf_cfg.get("rms_norm_eps", model_args.norm_eps)

    model_cls = Llama3Quantized if quantization == "w8a16" else Llama3
    model, report = build_model_from_weights(
        model_cls=model_cls,
        model_args=model_args,
        weights_path=weights_path,
        device=device,
        dtype=torch.bfloat16,
        quantization=quantization,
    )
    return model, model_args, report


def _extract_layer_outputs(model: Llama3, tokens: torch.Tensor) -> Tuple[torch.Tensor, list]:
    """Run forward and capture the output of each TransformerBlock."""
    hooks = []
    layer_outputs = []

    def hook_fn(module, inp, out):
        layer_outputs.append(out.detach().clone())

    for layer in model.layers:
        hooks.append(layer.register_forward_hook(hook_fn))

    try:
        with torch.no_grad():
            logits = model(tokens, return_all_logits=True)
    finally:
        for h in hooks:
            h.remove()

    return logits, layer_outputs


def _compute_perplexity(model: Llama3, tokens: torch.Tensor) -> float:
    with torch.no_grad():
        logits = model(tokens, return_all_logits=True)
    return perplexity(logits, tokens)


def _generate_text(model: Llama3, tokenizer: Tokenizer, prompt_text: str, max_new_tokens: int) -> str:
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    generated = list(input_ids)

    with torch.no_grad():
        prefill_tokens = torch.tensor([generated], dtype=torch.long, device=device)
        logits = model(prefill_tokens, start_pos=0)
        current_pos = prefill_tokens.size(1)

        input_token_tensor = torch.zeros((1, 1), dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            next_token = sample(logits, temperature=0.0, top_p=1.0)
            next_id = int(next_token.item())
            generated.append(next_id)
            if tokenizer.eos_id is not None and next_id == tokenizer.eos_id:
                break
            input_token_tensor[0, 0] = next_id
            logits = model(input_token_tensor, start_pos=current_pos)
            current_pos += 1

    return tokenizer.decode(generated[len(input_ids):])


def _build_chat_prompt(tokenizer: Tokenizer, system: str, user: str) -> str:
    inner = tokenizer.tokenizer
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return inner.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def main():
    parser = argparse.ArgumentParser(description="W8A16 quantization accuracy validation")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--text-file", default=None, help="Path to text file for perplexity evaluation")
    parser.add_argument("--prompt", default="Explain the importance of renewable energy in one paragraph.")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-eval-tokens", type=int, default=512, help="Max tokens to use for perplexity")
    args = parser.parse_args()

    env = _load_env(args.env_file)
    tokenizer_path = _resolve_option(args.tokenizer, "LLM_TOKENIZER", env, None, str)
    weights_path = _resolve_option(args.weights, "LLM_WEIGHTS", env, None, str)

    if not tokenizer_path or not weights_path:
        parser.error("--tokenizer/--weights (or LLM_TOKENIZER/LLM_WEIGHTS) are required.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("W8A16 QUANTIZATION ACCURACY VALIDATION")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Weights: {weights_path}")
    print("=" * 70)

    tokenizer = Tokenizer(tokenizer_path)

    print("\n[1] Loading bf16 baseline model...")
    t0 = time.time()
    bf16_model, bf16_args, bf16_report = _load_model(weights_path, tokenizer_path, device, None, args.max_seq_len)
    print(f"    Loaded {bf16_report['loaded']} tensors in {time.time() - t0:.2f}s")

    print("\n[2] Loading W8A16 quantized model...")
    t0 = time.time()
    quant_model, quant_args, quant_report = _load_model(weights_path, tokenizer_path, device, "w8a16", args.max_seq_len)
    print(f"    Loaded {quant_report['loaded']} tensors in {time.time() - t0:.2f}s")

    # --- Layer-level similarity ---
    print("\n[3] Layer-level cosine similarity (same prompt)...")
    prompt_text = _build_chat_prompt(tokenizer, args.system_prompt, args.prompt)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    eval_ids = prompt_ids[: args.max_eval_tokens]
    tokens = torch.tensor([eval_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        bf16_logits, bf16_layers = _extract_layer_outputs(bf16_model, tokens)
        quant_logits, quant_layers = _extract_layer_outputs(quant_model, tokens)

    layer_sims = []
    for idx, (bf16_out, quant_out) in enumerate(zip(bf16_layers, quant_layers)):
        sim = cosine_similarity(bf16_out, quant_out)
        layer_sims.append(sim.item())
        print(f"    Layer {idx:2d}: cos_sim={sim.item():.6f}")
    min_sim = min(layer_sims)
    mean_sim = sum(layer_sims) / len(layer_sims)
    print(f"    Mean cos_sim={mean_sim:.6f}, Min cos_sim={min_sim:.6f}")

    # --- Logit-level metrics ---
    print("\n[4] Logit-level metrics...")
    mse = ((bf16_logits.float() - quant_logits.float()) ** 2).mean().item()
    max_abs = logits_max_abs_error(bf16_logits, quant_logits)
    rel_err = logits_relative_error(bf16_logits, quant_logits)
    top1 = top1_agreement(bf16_logits, quant_logits)
    print(f"    Logits MSE:          {mse:.6e}")
    print(f"    Logits max abs err:  {max_abs:.6f}")
    print(f"    Logits relative err: {rel_err:.6e}")
    print(f"    Top-1 agreement:     {top1 * 100:.2f}%")

    # --- Perplexity ---
    print("\n[5] Perplexity comparison...")
    if args.text_file:
        text = Path(args.text_file).expanduser().read_text(encoding="utf-8")
        tokens_full = tokenizer.encode(text, add_special_tokens=False)[: args.max_eval_tokens]
        tokens_pt = torch.tensor([tokens_full], dtype=torch.long, device=device)
    else:
        tokens_pt = tokens

    bf16_ppl = _compute_perplexity(bf16_model, tokens_pt)
    quant_ppl = _compute_perplexity(quant_model, tokens_pt)
    ppl_increase = (quant_ppl - bf16_ppl) / bf16_ppl * 100
    print(f"    BF16  PPL: {bf16_ppl:.4f}")
    print(f"    W8A16 PPL: {quant_ppl:.4f}")
    print(f"    Relative increase: {ppl_increase:+.3f}%")

    # --- Generation comparison ---
    print("\n[6] Greedy generation comparison...")
    bf16_text = _generate_text(bf16_model, tokenizer, prompt_text, args.max_new_tokens)
    quant_text = _generate_text(quant_model, tokenizer, prompt_text, args.max_new_tokens)
    print(f"    BF16:  {bf16_text[:500]}{'...' if len(bf16_text) > 500 else ''}")
    print(f"    W8A16: {quant_text[:500]}{'...' if len(quant_text) > 500 else ''}")

    # --- Pass/fail summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    checks = {
        "Mean layer cos_sim >= 0.99": mean_sim >= 0.99,
        "Top-1 agreement >= 90%": top1 >= 0.90,
        "PPL increase <= 1%": ppl_increase <= 1.0,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    all_ok = all(checks.values())
    print(f"\nOverall: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
