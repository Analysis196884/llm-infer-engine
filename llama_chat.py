import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = (
    "/home/analysis/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).cuda()

dialogs = [
    {"role": "user", "content": "Hello, who are you?"}
]

input_ids = tokenizer.apply_chat_template(
    dialogs, add_generation_prompt=True, return_tensors="pt"
)["input_ids"].cuda()
attention_mask = torch.ones_like(input_ids)

print("\nModel Response:")

outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=256,
    temperature=0.6,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id,
)

print(tokenizer.decode(outputs[0][input_ids.shape[-1] :], skip_special_tokens=True))
