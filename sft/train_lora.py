# sft/train_lora.py
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

os.makedirs("./lora_model", exist_ok=True)

base_model = "Qwen/Qwen2.5-1.5B-Instruct"

# ===== 1. 4bit量化 =====
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# ===== 2. tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    trust_remote_code=True
)

# ===== 3. 模型 =====
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# ===== 4. LoRA配置 =====
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ===== 5. 加载数据 =====
dataset = load_dataset(
    "json",
    data_files={"train": "../data/processed/train.jsonl"}  # ⚠️ 建议用jsonl
)

# ===== 6. 预处理（🔥核心）=====
def preprocess(example):
    messages = example["messages"]

    # 👉 使用Qwen官方chat模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding="max_length"
    )

    input_ids = tokenized["input_ids"]
    labels = input_ids.copy()

    # ===== 🔥关键：mask掉user部分 =====
    # 思路：只让assistant参与loss
    assistant_start = text.rfind("<|im_start|>assistant")
    if assistant_start != -1:
        assistant_text = text[assistant_start:]
        assistant_ids = tokenizer(
            assistant_text,
            truncation=True,
            max_length=512
        )["input_ids"]

        # 前面mask掉
        mask_len = len(input_ids) - len(assistant_ids)
        labels[:mask_len] = [-100] * mask_len

    tokenized["labels"] = labels
    return tokenized


dataset = dataset.map(
    preprocess,
    remove_columns=dataset["train"].column_names
)

# ===== 7. 训练参数 =====
training_args = TrainingArguments(
    output_dir="./lora_model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,   # 👉 等效batch=4
    num_train_epochs=3,
    learning_rate=2e-4,
    save_strategy="epoch",
    logging_steps=10,
    fp16=True,
    optim="paged_adamw_32bit",
    report_to="none"
)

# ===== 8. Trainer =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

# ===== 9. 开始训练 =====
trainer.train()

# ===== 10. 保存最终模型 =====
model.save_pretrained("./lora_model_qa")
tokenizer.save_pretrained("./lora_model_qa")

print("✅ LoRA training finished!")