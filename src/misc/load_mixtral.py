import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# https://github.com/huggingface/blog/blob/main/mixtral.md
messages = [
    {"role": "user", "content": "Explain what a Mixture of Experts is in less than 100 words."},
]
prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompts)
inputs = tokenizer(prompts, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
