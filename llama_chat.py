import argparse
from pathlib import Path
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer


def main():
    parser = argparse.ArgumentParser(description="Llama chat")
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    args = parser.parse_args()

    env = {}
    if Path(".env").exists():
        env = {
            k: v.strip('"')
            for k, v in (
                l.split("=", 1)
                for l in Path(".env").read_text().splitlines()
                if "=" in l
            )
        }

    model = (
        args.model
        or env.get("LLM_MODEL")
        or "/home/analysis/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct"
    )
    temp = args.temperature or env.get("LLM_TEMPERATURE") or "0.6"
    top = args.top_p or env.get("LLM_TOP_P") or "0.9"
    max_tokens = args.max_new_tokens or env.get("LLM_MAX_NEW_TOKENS") or "256"

    prompt = args.prompt or env.get("LLM_PROMPT")
    if args.prompt_file:
        prompt = Path(args.prompt_file).expanduser().read_text()

    if not prompt:
        raise ValueError(
            "Missing prompt: use --prompt, --prompt-file, or set LLM_PROMPT in .env"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("LLAMA CHAT (HF BASELINE)")
    print("=" * 70)
    print(f"Device: {device}")

    print(f"\n[1] Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16)
    model = model.to(device)
    print(f"    {time.time() - t0:.2f}s")

    messages = [
        {
            "role": "system",
            "content": env.get("LLM_SYSTEM_PROMPT", "You are a helpful assistant."),
        },
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = input_ids["input_ids"].to(device)
    attention_mask = torch.ones_like(input_ids)
    prompt_len = input_ids.size(1)
    print(f"    Prompt length: {prompt_len} tokens")

    print(f"\n[2] Generating...")
    print()
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    torch.cuda.synchronize()
    gen_start = time.time()
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(max_tokens),
        temperature=float(temp),
        top_p=float(top),
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    torch.cuda.synchronize()
    total_time = time.time() - gen_start
    new_tokens = outputs.size(1) - prompt_len

    # Prefill timing (warmed up by the full generation above)
    torch.cuda.synchronize()
    prefill_start = time.time()
    _ = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1,
        temperature=float(temp),
        top_p=float(top),
        pad_token_id=tokenizer.eos_token_id,
        streamer=None,
    )
    torch.cuda.synchronize()
    prefill_time = time.time() - prefill_start

    decode_time = total_time - prefill_time
    tps = new_tokens / max(decode_time, 1e-6)

    print(f"\n{'=' * 70}")
    print(f"    Prefill: {prefill_time:.4f}s ({prompt_len} tokens)")
    print(f"    Decode:  {new_tokens} tokens in {decode_time:.4f}s")
    print(f"    TPS:     {tps:.2f} tokens/s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
