import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM


@torch.inference_mode()
def main():
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    peft_id = "outputs"
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model = PeftModel.from_pretrained(model=model, model_id=peft_id)

    model.push_to_hub("hpprc/Mixtral-8x7B-Instruct-ja-en")


if __name__ == "__main__":
    main()
