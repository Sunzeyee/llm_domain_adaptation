# semantic_eval_clean.py

import json
import numpy as np
import faiss
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
BASE_PATH = "../data/index/novel_semantic"


# ===== 相似度 =====
def sim(a, b):
    return cosine_similarity([a], [b])[0][0]


# ===== GT（文本匹配，不跳过🔥）=====
def find_gt_doc_by_text(answer, docs):
    for i, doc in enumerate(docs):
        if answer in doc:
            return i
    return -1


# ===== 检索 =====
def retrieve(index, q, top_k=5):
    q_emb = model.encode([q])
    _, I = index.search(q_emb, top_k)
    return I[0]


# ===== RAG =====
def rag_answer(docs, ids):
    return "\n".join([docs[i] for i in ids])[:300]


# ===== 评估 =====
def evaluate(path, dataset):

    index = faiss.read_index(f"{path}/knowledge.index")
    docs = np.load(f"{path}/docs.npy", allow_pickle=True)

    recalls, sims = [], []

    for item in dataset:

        q = item["question"]
        answer = item["answer"]

        gt_id = find_gt_doc_by_text(answer, docs)

        retrieved_ids = retrieve(index, q)

        # ❗ 不跳过
        if gt_id == -1:
            r = 0
        else:
            r = int(gt_id in retrieved_ids)

        pred = rag_answer(docs, retrieved_ids)

        pred_emb = model.encode([pred])[0]
        gt_emb = model.encode([answer])[0]

        s = sim(pred_emb, gt_emb)

        recalls.append(r)
        sims.append(s)

    return np.mean(recalls), np.mean(sims)


# ===== 主流程 =====
def run():

    with open("../data/processed/novel_test.json", encoding="utf-8") as f:
        dataset = json.load(f)

    results = {"thresholds": [], "recalls": [], "sims": []}

    for t in THRESHOLDS:
        name = f"semantic_t{t}"
        path = f"{BASE_PATH}/{name}"

        print(f"🚀 正在评估: {name}")
        r, s = evaluate(path, dataset)

        results["thresholds"].append(t)
        results["recalls"].append(r)
        results["sims"].append(s)

        print(f"Recall={r:.3f}, Sim={s:.3f}\n")

    plot(results)


# ===== 画图 =====
def plot(results):

    plt.figure()

    plt.plot(results["thresholds"], results["recalls"], marker='o', label="Recall")
    plt.plot(results["thresholds"], results["sims"], marker='^', label="Similarity")

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Semantic Chunk Evaluation (Clean)")

    plt.legend()
    plt.grid()

    plt.savefig("../results/novel_semantic.png")
    plt.show()


if __name__ == "__main__":
    run()