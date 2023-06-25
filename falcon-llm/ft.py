"""
Fine-Tune Falcon LLM models
"""

import argparse
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer
from accelerate import Accelerator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--dataset_name", type=str, default="timdettmers/openassistant-guanaco")
    parser.add_argument("-m", "--model_name", type=str, default="ybelkada/falcon-7b-sharded-bf16")
    parser.add_argument("-q", "--quantize_mode", type=str, default="4bit", choices=["4bit", "8bit", "16bit"])
    parser.add_argument("-s", "--steps", type=int, default=50)
    parser.add_argument("-b", "--batch_size_per_device", type=int, default=16)

    return parser.parse_args()


def run_training(args):
    dataset_name = args.dataset_name
    dataset = load_dataset(dataset_name, split="train")

    model_name = args.model_name

    if args.quantize_mode == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif args.quantize_mode == "8bit":
            bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif args.quantize_mode == "16bit":
        bnb_config = BitsAndBytesConfig(
        )        

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": Accelerator().process_index},
        trust_remote_code=True
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    lora_alpha = 16
    lora_dropout = 0.1
    lora_r = 64

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ]
    )

    output_dir = "./results"
    per_device_train_batch_size = args.batch_size_per_device
    gradient_accumulation_steps = 1
    gradient_checkpointing = False
    optim = "paged_adamw_32bit"
    save_steps = args.steps
    logging_steps = 10
    learning_rate = 2e-4
    max_grad_norm = 0.3
    max_steps = args.steps
    warmup_ratio = 0.03
    lr_scheduler_type = "constant"

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        fp16=True,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=lr_scheduler_type,
        ddp_find_unused_parameters=False,
    )

    max_seq_length = 512

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    trainer.train()


def main(args):
    run_training(args)


if __name__ == "__main__":
    args = get_args()
    main(args)
