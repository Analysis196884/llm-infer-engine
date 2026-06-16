import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


def _resolve_str_option(cli_value, env_key: str, env: dict, default: str) -> str:
    if cli_value is not None:
        return str(cli_value)
    return env.get(env_key) or default


def _load_env(path: Path) -> dict:
    if not path.exists():
        return {}
    return {
        k: v.strip('"')
        for k, v in (l.split("=", 1) for l in path.read_text().splitlines() if "=" in l)
    }


def _build_prompt(args, env: dict) -> str:
    prompt = args.prompt or env.get("LLM_PROMPT")
    if args.prompt_file:
        prompt = Path(args.prompt_file).expanduser().read_text()
    if not prompt:
        raise ValueError(
            "Missing prompt: use --prompt, --prompt-file, or set LLM_PROMPT in .env"
        )
    return prompt


def _configure_model_for_greedy(model, temperature: float):
    if temperature <= 0:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None


def _build_gen_kwargs(temperature: float, top_p: float, **base_kwargs):
    if temperature > 0:
        return {**base_kwargs, "temperature": temperature, "top_p": top_p}
    return {**base_kwargs, "do_sample": False}


def main():
    parser = argparse.ArgumentParser(description="Llama chat (HF baseline)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    args = parser.parse_args()

    env = _load_env(Path(".env"))

    model_path = (
        args.model
        or env.get("LLM_MODEL")
        or "/home/analysis/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct"
    )
    temperature = float(_resolve_str_option(args.temperature, "LLM_TEMPERATURE", env, "0.0"))
    top_p = float(_resolve_str_option(args.top_p, "LLM_TOP_P", env, "1.0"))
    max_new_tokens = int(_resolve_str_option(args.max_new_tokens, "LLM_MAX_NEW_TOKENS", env, "256"))
    system_prompt = env.get("LLM_SYSTEM_PROMPT", "You are a helpful assistant.")

    prompt = _build_prompt(args, env)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("LLAMA CHAT (HF BASELINE)")
    print("=" * 70)
    print(f"Device: {device}")

    print("\n[1] Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    model = model.to(device)
    _configure_model_for_greedy(model, temperature)
    print(f"    {time.time() - t0:.2f}s")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = input_ids["input_ids"].to(device)
    attention_mask = torch.ones_like(input_ids)
    prompt_len = input_ids.size(1)
    print(f"    Prompt length: {prompt_len} tokens")

    print("\n[2] Generating...\n")
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    torch.cuda.synchronize()
    gen_start = time.time()
    outputs = model.generate(**_build_gen_kwargs(
        temperature, top_p,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    ))
    torch.cuda.synchronize()
    gen_time = time.time() - gen_start
    new_tokens = outputs.size(1) - prompt_len

    print(f"\n{'=' * 70}")
    print(f"    Generated: {new_tokens} tokens in {gen_time:.4f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
