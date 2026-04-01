# rag/hybrid_rag_inference.py

import os
import torch
import faiss
import numpy as np
import jieba

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from rank_bm25 import BM25Okapi


# ===== 1. LLM =====
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


# ===== 2. Embedding & Index =====
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

INDEX_PATH = "../data/index/chunk_300_overlap_50/knowledge.index"
DOCS_PATH = "../data/index/chunk_300_overlap_50/docs.npy"

index = faiss.read_index(INDEX_PATH)
docs = np.load(DOCS_PATH, allow_pickle=True)


# ===== 3. BM25 =====
def tokenize(text):
    return list(jieba.cut(text))

tokenized_docs = [tokenize(doc) for doc in docs]
bm25 = BM25Okapi(tokenized_docs)


# ===== 4. 工具函数 =====
def normalize(scores):
    min_v = np.min(scores)
    max_v = np.max(scores)
    if max_v - min_v < 1e-6:
        return np.zeros_like(scores)
    return (scores - min_v) / (max_v - min_v)


# ===== 5. Hybrid 检索（关键🔥）=====
def hybrid_retrieve(query, alpha=0.6, top_k=5, dense_k=100):

    # ===== Dense召回候选 =====
    q_emb = embed_model.encode([query])
    D, I = index.search(q_emb, dense_k)

    dense_ids = I[0]
    dense_scores = -D[0]  # L2 → 相似度
    dense_scores = normalize(dense_scores)

    # ===== BM25打分（只在候选集）=====
    tokenized_query = tokenize(query)
    bm25_all = np.array(bm25.get_scores(tokenized_query))

    bm25_scores = bm25_all[dense_ids]
    bm25_scores = normalize(bm25_scores)

    # ===== 融合 =====
    final_scores = alpha * bm25_scores + (1 - alpha) * dense_scores

    # ===== 排序 =====
    sorted_idx = np.argsort(final_scores)[::-1][:top_k]
    top_ids = dense_ids[sorted_idx]

    return top_ids


# ===== 6. RAG生成 =====
def rag_answer(question, k=5, alpha=0.6):

    # 🔥 Hybrid 检索
    retrieved_ids = hybrid_retrieve(question, alpha=alpha, top_k=k)

    context = "\n\n".join([docs[i] for i in retrieved_ids])

    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question: {question}
Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False  # 更稳定
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ===== 7. 测试 =====
if __name__ == "__main__":

    question = "你们的服务怎么做监控和告警？"

    print("Q:", question)
    print("\n===== HYBRID RAG =====")
    print(rag_answer(question, alpha=0.6))