# sft/inference.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base_model = "Qwen/Qwen2.5-1.5B-Instruct"

# 4bit量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto"
)

# 加载LoRA权重
model = PeftModel.from_pretrained(model, "./lora_model/checkpoint-45")

prompt = "Instruction: When did J-20 enter service?\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=50
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))