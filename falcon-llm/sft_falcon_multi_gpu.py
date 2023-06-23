import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from transformers import TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig


dataset_name = "timdettmers/openassistant-guanaco"
dataset = load_dataset(dataset_name, split="train")

model_name = "ybelkada/falcon-7b-sharded-bf16"
# model_name = "tiiuae/falcon-40b"
# model_name = "eachadea/vicuna-13b-1.1"
# model_name = "eachadea/vicuna-7b-1.1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
    trust_remote_code=True,
    device_map="auto",
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

training_arguments = TrainingArguments(
    output_dir = "./results",
    per_device_train_batch_size=1,
    fp16=True,
)

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

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=512,
    args=training_arguments,
    peft_config=peft_config,
)

for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

trainer.train()