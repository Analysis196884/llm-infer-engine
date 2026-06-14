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
    print(f"[1] Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16)
    model = model.to(device)
    print(f"    {time.time() - t0:.2f}s")

    print(f"[2] Generating...")
    t0 = time.time()
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
    
    print(f"\n[Result]:")
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(max_tokens),
        temperature=float(temp),
        top_p=float(top),
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    print(f"    {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
