import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2", 
    torch_dtype=torch.float32, 
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/phi-2",
    add_eos_token=True,
    add_bos_token=True,
    use_fast=False,
    trust_remote_code=True
)

inputs = tokenizer("Write a quicksort algorithm in Python.", return_tensors="pt")

outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)