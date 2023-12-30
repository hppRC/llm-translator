import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.inference_mode()
def main():
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    peft_id = "hpprc/Mixtral-8x7B-Instruct-ja-en"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=True,
    ).eval()

    model = PeftModel.from_pretrained(model=model, model_id=peft_id)

    messages = [
        # {"role": "user", "content": "Translate this English sentence into Japanese.\n" + input("En > ")},
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
    gen_ids = outputs[0][len(inputs.input_ids[0]):]
    out = tokenizer.decode(gen_ids, skip_special_tokens=True)
    out = out.split("\n")[0] # 生成しすぎることがあるので最初の一文だけ取り出すのがいいかも
    print(out)


if __name__ == "__main__":
    main()
