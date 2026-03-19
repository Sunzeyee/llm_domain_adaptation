# rag/rag_inference.py
import os
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 模型名称
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# 4bit量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# 加载嵌入模型和索引
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
index = faiss.read_index("knowledge.index")
docs = np.load("docs.npy", allow_pickle=True)

def rag_answer(question, k=2):
    q_emb = embed_model.encode([question], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    context = "\n".join([docs[i] for i in I[0]])

    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question: {question}
Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    question = "When did J-20 enter service?"
    print("Q:", question)
    print("A:", rag_answer(question))