import torch
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# ===== 基座模型 =====
base_model = "Qwen/Qwen2.5-1.5B-Instruct"

# ===== 4bit配置 =====
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto"
)

print("Loading LoRA...")
model = PeftModel.from_pretrained(
    model,
    "../sft/lora_model/checkpoint-336"
)

model.eval()


# ===== embedding模型 =====
embed_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

print("Loading FAISS index...")
index = faiss.read_index("../rag/knowledge.index")

docs = np.load(
    "../rag/docs.npy",
    allow_pickle=True
)


# ===== RAG函数 =====

def rag_lora_answer(question, k=4):

    q_emb = embed_model.encode(
        [question],
        convert_to_numpy=True
    )

    D, I = index.search(q_emb, k)

    context = "\n".join(
        [docs[i] for i in I[0]]
    )

    prompt = f"""Instruction: {question}

Information:
{context}

Answer:"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
    )

    generated = outputs[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True
    )


# ===== 测试 =====

if __name__ == "__main__":

    q = "Java语言有哪些特点？"

    print(q, rag_lora_answer(q))
