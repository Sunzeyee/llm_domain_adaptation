import json
import numpy as np
import faiss
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# ===== 参数 =====
THRESHOLDS = [0.6, 0.7, 0.75, 0.8, 0.85]
OVERLAPS = [0, 50]

BASE_PATH = "../data/index"


# ===== 相似度 =====
def sim(a, b):
    return cosine_similarity([a], [b])[0][0]


# ===== GT =====
def find_gt(answer_emb, doc_emb):
    sims = cosine_similarity([answer_emb], doc_emb)[0]
    return int(np.argmax(sims))


# ===== 检索 =====
def retrieve(index, q):
    q_emb = model.encode([q])
    _, I = index.search(q_emb, 5)
    return I[0]


# ===== RAG =====
def rag_answer(docs, ids):
    return "\n".join([docs[i] for i in ids])[:300]


# ===== 单配置 =====
def evaluate(path, dataset, answer_embs):

    index = faiss.read_index(f"{path}/knowledge.index")
    docs = np.load(f"{path}/docs.npy", allow_pickle=True)

    doc_embs = model.encode(docs, batch_size=64)

    recalls, sims = [], []

    for i, item in enumerate(dataset):

        gt_emb = answer_embs[i]
        gt_id = find_gt(gt_emb, doc_embs)

        ids = retrieve(index, item["question"])

        r = int(gt_id in ids)

        pred = rag_answer(docs, ids)
        pred_emb = model.encode([pred])[0]

        s = sim(pred_emb, gt_emb)

        recalls.append(r)
        sims.append(s)

    return np.mean(recalls), np.mean(sims)


# ===== 主流程 =====
def run():

    with open("../data/processed/test.json", encoding="utf-8") as f:
        dataset = json.load(f)

    answers = [d["answer"] for d in dataset]
    answer_embs = model.encode(answers, batch_size=32)

    data = {
        0: {"t": [], "recall": [], "sim": []},
        50: {"t": [], "recall": [], "sim": []}
    }

    for o in OVERLAPS:
        for t in THRESHOLDS:

            name = f"semantic_window_t{t}_overlap_{o}"
            path = f"{BASE_PATH}/{name}"

            print(f"🚀 {name}")

            r, s = evaluate(path, dataset, answer_embs)

            data[o]["t"].append(t)
            data[o]["recall"].append(r)
            data[o]["sim"].append(s)

            print(f"Recall={r:.3f}, Sim={s:.3f}")

    plot(data)


# ===== 画图 =====
def plot(data):

    plt.figure()

    colors = {
        "recall": ["#1f77b4", "#6baed6"],
        "sim": ["#d62728", "#fb6a4a"]
    }

    # Recall
    plt.plot(data[0]["t"], data[0]["recall"],
             marker='o', color=colors["recall"][0],
             label="Recall (overlap=0)")

    plt.plot(data[50]["t"], data[50]["recall"],
             marker='o', linestyle='--', color=colors["recall"][1],
             label="Recall (overlap=50)")

    # Similarity
    plt.plot(data[0]["t"], data[0]["sim"],
             marker='^', color=colors["sim"][0],
             label="Sim (overlap=0)")

    plt.plot(data[50]["t"], data[50]["sim"],
             marker='^', linestyle='--', color=colors["sim"][1],
             label="Sim (overlap=50)")

    plt.xlabel("Semantic Threshold")
    plt.ylabel("Score")
    plt.title("Semantic Window Chunking Threshold Experiment")

    plt.legend()
    plt.grid()

    plt.savefig("../results/semantic_threshold_curve.png")
    plt.show()


if __name__ == "__main__":
    run()