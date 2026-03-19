# sft/train_lora.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

os.makedirs("./lora_model", exist_ok=True)

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

# LoRA配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 加载训练数据
dataset = load_dataset("json", data_files={"train": "../data/train.json"})
def preprocess(example):

    text = f"Instruction: {example['instruction']}\nAnswer: {example['output']}"

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding="max_length"
    )

    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


dataset = load_dataset(
    "json",
    data_files={"train": "../data/train.json"}
)

dataset = dataset.map(preprocess)

# 训练参数
training_args = TrainingArguments(
    output_dir="./lora_model",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_steps=10,
    fp16=True,
    optim="paged_adamw_32bit"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

trainer.train()
print("✅ LoRA training finished!")