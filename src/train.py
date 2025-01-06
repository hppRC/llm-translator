import os
from dataclasses import dataclass, field

import bitsandbytes as bnb
import torch
import torch.utils.checkpoint
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    MistralForCausalLM,
    TrainingArguments,
)
from trl import SFTTrainer

import datasets as ds


@dataclass
class Args(TrainingArguments):
    output_dir: str = "outputs"
    model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    learning_rate: float = 1e-4
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 8
    warmup_ratio: float = 0.1

    max_seq_len: int = 1024
    weight_decay: float = 0.01

    lora_r: int = 16
    neftune_noise_alpha: float = 5.0

    logging_steps: int = 10

    eval_steps: int = 100
    evaluation_strategy: str = "steps"

    bf16: bool = True
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict = field(default_factory=lambda: {"use_reentrant": True})

    report_to: str = "none"
    ddp_find_unused_parameters: bool = False
    load_best_model_at_end: bool = False  # This is importnant for preventing hangup
    remove_unused_columns: bool = False

    optim: str = "paged_adamw_8bit"

    @property
    def lora_alpha(self) -> float:
        return self.lora_r * 2


def find_all_linear_names(model: nn.Module) -> list[str]:
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")

    return sorted(list(lora_module_names))


def main(args: Args):
    dataset: ds.Dataset = ds.load_from_disk("./datasets/en-ja-alignment").shuffle()
    datasets = dataset.train_test_split(test_size=1024)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model: MistralForCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        load_in_4bit=True,
        quantization_config=quantization_config,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs,
    )

    target_modules = find_all_linear_names(model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
        task_type="CAUSAL_LM",
    )

    peft_model: PeftModel = get_peft_model(model, lora_config)

    for name, module in peft_model.named_modules():
        if isinstance(module, LoraLayer):
            module.to(torch.bfloat16)
        elif "norm" in name:
            module.to(torch.bfloat16)
        elif "lm_head" in name or isinstance(module, nn.Embedding):
            module.to(torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="right")
    tokenizer.pad_token = tokenizer.unk_token

    def formatting_func(examples: dict[str, list]):
        output_texts = []

        for text_en, text_ja in zip(examples["en"], examples["ja"], strict=True):
            messages = [
                {"role": "user", "content": "Translate this Japanese sentence into English.\n" + text_ja},
                {"role": "assistant", "content": text_en},
            ]
            output_texts.append(tokenizer.apply_chat_template(messages, tokenize=False))

            messages = [
                {"role": "user", "content": "Translate this English sentence into Japanese.\n" + text_en},
                {"role": "assistant", "content": text_ja},
            ]
            output_texts.append(tokenizer.apply_chat_template(messages, tokenize=False))

        return output_texts

    trainer = SFTTrainer(
        args=args,
        model=peft_model,
        tokenizer=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        formatting_func=formatting_func,
        max_seq_length=args.max_seq_len,
    )

    trainer.train()
    trainer.save_model()
    trainer.state.save_to_json(os.path.join(args.output_dir, "trainer-state.json"))


if __name__ == "__main__":
    parser = HfArgumentParser(Args)
    args: Args = parser.parse_args_into_dataclasses()[0]
    main(args)
