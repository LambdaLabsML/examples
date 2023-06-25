import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


model = "tiiuae/falcon-7b"
# model = "tiiuae/falcon-40b"
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=0)
pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=0,
)

sequences = pipeline(
        "To make the perfect chocolate chip cookies,",
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
)

for seq in sequences:
    print(f"Result: {seq['generated_text']}")

