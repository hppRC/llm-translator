import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.inference_mode()
def main():
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    peft_id = "outputs"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=True,
        attn_implementation="flash_attention_2",
    ).eval()

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    model = PeftModel.from_pretrained(model=model, model_id=peft_id)

    while True:
        print("=" * 80)
        messages = [
            {"role": "user", "content": "Translate this Japanese sentence into English.\n" + input("Ja > ")},
        ]
        prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompts, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            top_p=0.95,
            num_beams=5,
            pad_token_id=tokenizer.eos_token_id,
        )
        print()
        print(tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=False).split("\n")[0])

        print()
        print("-" * 80)
        print()

        messages = [
            {"role": "user", "content": "Translate this English sentence into Japanese.\n" + input("En > ")},
        ]
        prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompts, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            top_p=0.95,
            num_beams=5,
            pad_token_id=tokenizer.eos_token_id,
        )
        print()
        print(tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=False).split("\n")[0])


if __name__ == "__main__":
    main()
