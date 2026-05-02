import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ===== 模型 =====
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# ===== embedding =====
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# ===== QA index =====
index = faiss.read_index("../data/index/qa_index/qa.index")
answers = np.load("../data/index/qa_index/answers.npy", allow_pickle=True)

# （可选）加载问题文本方便打印
import json
with open("../data/processed/test.json", encoding="utf-8") as f:
    dataset = json.load(f)
questions = [item["question"] for item in dataset]


def rag_answer(question, k=3):

    q_emb = embed_model.encode([question])
    D, I = index.search(q_emb, k)

    retrieved_q = [questions[i] for i in I[0]]
    retrieved_a = [answers[i] for i in I[0]]

    # ===== context =====
    context = ""
    for i in range(k):
        context += f"Q: {retrieved_q[i]}\nA: {retrieved_a[i]}\n\n"

    print("\n===== RETRIEVED CONTEXT =====")

    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question: {question}
Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=150)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    q = "什么是索引引擎层"
    print("Q:", q)
    print("A:", rag_answer(q))