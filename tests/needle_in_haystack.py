import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import ModelArgs
from src.kv_cache import KVCache
from src.loader import build_model_from_weights
from src.model import Llama3
from src.model_quantized import Llama3Quantized
from src.sampler import sample
from src.tokenizer import Tokenizer


DEFAULT_FILLER = """\
The field of artificial intelligence continues to evolve at a rapid pace. \
Researchers explore new architectures to improve model performance. \
Large language models are trained on vast amounts of text data. \
Optimization algorithms help minimize loss functions during training. \
Data quality is often more important than model size. \
Evaluation metrics guide the development of better systems. \
Interpretability remains a key challenge in modern machine learning. \
Robustness to distribution shift is critical for real-world deployment. \
"""

DEFAULT_NEEDLE = "The special magic number is 73928461."
DEFAULT_QUESTION = "What is the special magic number? Reply with only the number."
DEFAULT_ANSWER = "73928461"


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


def _parse_list(text: str, cast) -> List[int]:
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def _build_haystack(tokenizer: Tokenizer, target_tokens: int, filler_text: str) -> str:
    filler_ids = tokenizer.encode(filler_text, add_special_tokens=False)
    if not filler_ids:
        raise ValueError("Filler text is empty after tokenization")
    repeats = (target_tokens // len(filler_ids)) + 1
    repeated_ids = (filler_ids * repeats)[:target_tokens]
    return tokenizer.decode(repeated_ids)


def _insert_needle(
    haystack: str,
    needle: str,
    depth_percent: int,
    tokenizer: Tokenizer,
) -> str:
    ids = tokenizer.encode(haystack, add_special_tokens=False)
    needle_ids = tokenizer.encode(" " + needle, add_special_tokens=False)
    insert_pos = int(len(ids) * depth_percent / 100)
    insert_pos = max(0, min(insert_pos, len(ids)))
    combined_ids = ids[:insert_pos] + needle_ids + ids[insert_pos:]
    return tokenizer.decode(combined_ids)


def _inner_tokenizer(tokenizer):
    return tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer


def _configure_tokenizer(tokenizer):
    inner = _inner_tokenizer(tokenizer)
    if hasattr(inner, "clean_up_tokenization_spaces"):
        inner.clean_up_tokenization_spaces = False


def _build_chat_prompt(tokenizer, system: str, context: str, question: str) -> str:
    inner = _inner_tokenizer(tokenizer)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    return inner.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def _generate_custom(
    model,
    tokenizer: Tokenizer,
    model_args: ModelArgs,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_len = len(input_ids)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    kv_cache = KVCache(model_args, dtype=dtype)

    generated = list(input_ids)
    current_pos = 0
    input_token_tensor = torch.zeros((1, 1), dtype=torch.long, device=device)

    with torch.no_grad():
        prefill_tokens = torch.tensor([input_ids], dtype=torch.long, device=device)
        logits = model(prefill_tokens, start_pos=0, kv_cache=kv_cache)
        current_pos = prefill_tokens.size(1)

        for _ in range(max_new_tokens):
            next_token = sample(logits, temperature=temperature, top_p=top_p)
            next_id = int(next_token.item())
            generated.append(next_id)

            if tokenizer.eos_id is not None and next_id == tokenizer.eos_id:
                break
            if current_pos >= model_args.max_seq_len:
                break

            input_token_tensor[0, 0] = next_id
            logits = model(input_token_tensor, start_pos=current_pos, kv_cache=kv_cache)
            current_pos += 1

    return tokenizer.decode(generated[prompt_len:])


def _generate_hf(
    model,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    kwargs = {"max_new_tokens": max_new_tokens, "pad_token_id": tokenizer.eos_token_id}
    if temperature > 0:
        kwargs.update(temperature=temperature, top_p=top_p)
    else:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        kwargs["do_sample"] = False

    outputs = model.generate(**inputs, **kwargs)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def _run_single_test(
    model,
    tokenizer,
    model_args: Optional[ModelArgs],
    context_length: int,
    depth_percent: int,
    needle: str,
    question: str,
    expected_answer: str,
    system_prompt: str,
    filler_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    backend: str,
    device: str,
) -> dict:
    haystack = _build_haystack(tokenizer, context_length, filler_text)
    context = _insert_needle(haystack, needle, depth_percent, tokenizer)
    prompt_text = _build_chat_prompt(tokenizer, system_prompt, context, question)
    actual_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))

    torch.cuda.synchronize()
    start = time.time()
    if backend == "custom":
        answer = _generate_custom(
            model, tokenizer, model_args, prompt_text, max_new_tokens, temperature, top_p
        )
    else:
        answer = _generate_hf(
            model, tokenizer, prompt_text, max_new_tokens, temperature, top_p, device
        )
    torch.cuda.synchronize()
    elapsed = time.time() - start

    correct = expected_answer.lower().strip() in answer.lower().strip()
    return {
        "context_length": context_length,
        "actual_tokens": actual_tokens,
        "depth_percent": depth_percent,
        "answer": answer.strip(),
        "correct": correct,
        "time": elapsed,
    }


def _load_custom_model(
    weights_path: str,
    tokenizer_path: str,
    device: str,
    max_seq_len: int,
    quantization: Optional[str] = None,
):
    hf_cfg = {}
    cfg_path = Path(tokenizer_path) / "config.json"
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


def _load_hf_model(model_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    model = model.to(device)
    return model, tokenizer


def _print_results(results: List[dict], context_lengths: List[int], depths: List[int]):
    print("\n" + "=" * 70)
    print("NEEDLE IN A HAYSTACK RESULTS")
    print("=" * 70)

    max_len_width = max(len(str(x)) for x in context_lengths)
    header = "depth\\len".rjust(12) + " | " + " | ".join(str(x).rjust(max_len_width) for x in context_lengths)
    print(header)
    print("-" * len(header))

    score_grid = {}
    for r in results:
        score_grid[(r["depth_percent"], r["context_length"])] = "Y" if r["correct"] else "N"

    for d in depths:
        row = f"{d}%".rjust(12) + " | "
        row += " | ".join(score_grid.get((d, L), "?").rjust(max_len_width) for L in context_lengths)
        print(row)

    total = len(results)
    correct = sum(r["correct"] for r in results)
    print("-" * len(header))
    print(f"Accuracy: {correct}/{total} ({100 * correct / total:.1f}%)")
    print(f"Avg time: {sum(r['time'] for r in results) / total:.2f}s")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Needle in a Haystack long-context evaluation")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--backend", default="custom", choices=["custom", "hf"])
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--context-lengths", default="512,1024,2048", type=str)
    parser.add_argument("--depths", default="0,25,50,75,100", type=str)
    parser.add_argument("--needle", default=DEFAULT_NEEDLE)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--expected-answer", default=DEFAULT_ANSWER)
    parser.add_argument("--system-prompt", default="You are a helpful assistant. Answer the question using only the provided context. Be concise.")
    parser.add_argument("--filler-text", default=None, help="Path to a text file used as haystack filler")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--quantization", type=str, default=None, choices=["w8a16"], help="Quantization for custom backend")
    parser.add_argument("--output", default=None, help="Path to save results JSON")
    args = parser.parse_args()

    env = _load_env(args.env_file)
    tokenizer_path = _resolve_option(args.tokenizer, "LLM_TOKENIZER", env, None, str)
    weights_path = _resolve_option(args.weights, "LLM_WEIGHTS", env, "", str)
    model_path = _resolve_option(args.model, "LLM_MODEL", env, tokenizer_path, str)

    if not model_path:
        parser.error("Missing model/tokenizer path. Provide --tokenizer/--model or set LLM_MODEL/LLM_TOKENIZER.")
    if args.backend == "custom" and not weights_path:
        parser.error("Custom backend requires --weights or LLM_WEIGHTS.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    context_lengths = _parse_list(args.context_lengths, int)
    depths = _parse_list(args.depths, int)

    filler_text = DEFAULT_FILLER
    if args.filler_text:
        filler_text = Path(args.filler_text).expanduser().read_text(encoding="utf-8")

    print("=" * 70)
    print("NEEDLE IN A HAYSTACK")
    print("=" * 70)
    print(f"Backend: {args.backend}")
    print(f"Device: {device}")
    print(f"Quantization: {args.quantization or 'none'}")
    print(f"Context lengths: {context_lengths}")
    print(f"Depths: {depths}%")
    print(f"Needle: {args.needle}")
    print(f"Question: {args.question}")
    print("=" * 70)

    print("\n[1] Loading model...")
    t0 = time.time()
    if args.backend == "custom":
        model, model_args, report = _load_custom_model(
            weights_path,
            tokenizer_path or model_path,
            device,
            args.max_seq_len,
            quantization=args.quantization,
        )
        tokenizer = Tokenizer(tokenizer_path or model_path)
        _configure_tokenizer(tokenizer)
        print(f"    Loaded {report['loaded']} tensors in {time.time() - t0:.2f}s")
    else:
        model, tokenizer = _load_hf_model(model_path, device)
        _configure_tokenizer(tokenizer)
        print(f"    Loaded HF model in {time.time() - t0:.2f}s")
    model_args = None if args.backend == "hf" else model_args

    print("\n[2] Running tests...")
    results = []
    for context_length in context_lengths:
        for depth in depths:
            print(f"    context={context_length}, depth={depth}% ... ", end="", flush=True)
            result = _run_single_test(
                model=model,
                tokenizer=tokenizer,
                model_args=model_args,
                context_length=context_length,
                depth_percent=depth,
                needle=args.needle,
                question=args.question,
                expected_answer=args.expected_answer,
                system_prompt=args.system_prompt,
                filler_text=filler_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                backend=args.backend,
                device=device,
            )
            results.append(result)
            print(f"{'PASS' if result['correct'] else 'FAIL'} ({result['actual_tokens']} tokens, {result['time']:.2f}s)")

    _print_results(results, context_lengths, depths)

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
