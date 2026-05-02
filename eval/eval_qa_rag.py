import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ===== 模型 =====
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# ===== QA索引 =====
index = faiss.read_index("../data/index/qa_index/qa.index")
answers = np.load("../data/index/qa_index/answers.npy", allow_pickle=True)


# ===== 相似度 =====
def similarity_vec(a_vec, b_vec):
    return cosine_similarity([a_vec], [b_vec])[0][0]


# ===== 检索 =====
def retrieve(query, top_k=5):
    q_emb = embed_model.encode([query])
    D, I = index.search(q_emb, top_k)
    return I[0]


# ===== 找GT（语义版🔥）=====
def find_gt(answer_emb, answer_embeddings):
    sims = cosine_similarity([answer_emb], answer_embeddings)[0]
    return int(np.argmax(sims))


# ===== RAG =====
def rag_answer(retrieved_ids):
    context = "\n".join([answers[i] for i in retrieved_ids])
    return context[:300]


# ===== 主评估 =====
def run_eval():

    with open("../data/processed/test.json", encoding="utf-8") as f:
        dataset = json.load(f)

    # ===== 预计算 =====
    answer_texts = [item["answer"] for item in dataset]
    answer_embeddings = embed_model.encode(answer_texts, batch_size=32)

    recalls = []
    sims = []

    for idx, item in enumerate(dataset):

        q = item["question"]
        gt_emb = answer_embeddings[idx]

        # ===== GT id =====
        gt_id = find_gt(gt_emb, answer_embeddings)

        # ===== 检索 =====
        retrieved_ids = retrieve(q, top_k=5)

        # ===== Recall =====
        r = int(gt_id in retrieved_ids)

        # ===== 生成 =====
        pred = rag_answer(retrieved_ids)
        pred_emb = embed_model.encode([pred])[0]

        sim = similarity_vec(pred_emb, gt_emb)

        recalls.append(r)
        sims.append(sim)

        print("\n========================")
        print(f"Q: {q}")
        print(f"GT ID: {gt_id}")
        print(f"Retrieved: {retrieved_ids}")
        print(f"Recall@5: {r}")
        print(f"Similarity: {sim:.3f}")

    print("\n===== FINAL RESULT =====")
    print("Recall@5:", np.mean(recalls))
    print("Avg Similarity:", np.mean(sims))


if __name__ == "__main__":
    run_eval()