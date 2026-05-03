import json
import numpy as np
import faiss
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ===== 模型 =====
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# ===== 配置 =====
CONFIGS = [
    "chunk_100_overlap_0",
    "chunk_100_overlap_50",
    "chunk_200_overlap_0",
    "chunk_200_overlap_50",
    "chunk_300_overlap_0",
    "chunk_300_overlap_50",
    "chunk_400_overlap_0",
    "chunk_400_overlap_50",
    "chunk_500_overlap_0",
    "chunk_500_overlap_50",
]

BASE_PATH = "../data/index/novel_chunk"


# ===== 相似度 =====
def similarity_vec(a, b):
    return cosine_similarity([a], [b])[0][0]


# ===== ✅ GT（文本匹配）=====
def find_gt_doc_by_text(answer, docs):
    for i, doc in enumerate(docs):
        if answer in doc:
            return i
    return -1


# ===== 检索 =====
def retrieve(index, query, top_k=5):
    q_emb = embed_model.encode([query])
    _, I = index.search(q_emb, top_k)
    return I[0]


# ===== RAG =====
def rag_answer(docs, ids):
    return "\n".join([docs[i] for i in ids])[:300]


# ===== 评估 =====
def evaluate_config(config_name, dataset):

    print(f"\n🚀 Running config: {config_name}")

    index = faiss.read_index(f"{BASE_PATH}/{config_name}/knowledge.index")
    docs = np.load(f"{BASE_PATH}/{config_name}/docs.npy", allow_pickle=True)

    recalls, sims = [], []

    for item in dataset:

        q = item["question"]
        answer = item["answer"]

        # ===== GT =====
        gt_id = find_gt_doc_by_text(answer, docs)

        # ===== 检索 =====
        ids = retrieve(index, q)

        # ===== ❗ 改这里：不跳过 =====
        if gt_id == -1:
            r = 0
        else:
            r = int(gt_id in ids)

        # ===== 生成 =====
        pred = rag_answer(docs, ids)

        pred_emb = embed_model.encode([pred])[0]
        gt_emb = embed_model.encode([answer])[0]

        s = similarity_vec(pred_emb, gt_emb)

        recalls.append(r)
        sims.append(s)

    return np.mean(recalls), np.mean(sims)


# ===== 主流程 =====
def run_all():

    with open("../data/processed/novel_test.json", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []

    for config in CONFIGS:
        r, s = evaluate_config(config, dataset)
        results.append({"config": config, "recall": r, "sim": s})

    print("\n===== ALL RESULTS =====")
    for r in results:
        print(r)

    plot_results(results)


# ===== 可视化 =====
def plot_results(results):

    data = {
        0: {"size": [], "recall": [], "sim": []},
        50: {"size": [], "recall": [], "sim": []}
    }

    for r in results:
        name = r["config"]
        size = int(name.split("_")[1])
        overlap = int(name.split("_")[-1])

        data[overlap]["size"].append(size)
        data[overlap]["recall"].append(r["recall"])
        data[overlap]["sim"].append(r["sim"])

    for overlap in data:
        idx = np.argsort(data[overlap]["size"])
        for key in data[overlap]:
            data[overlap][key] = np.array(data[overlap][key])[idx]

    plt.figure()

    # 新增：颜色设置
    colors = {
        "recall": ["#1f77b4", "#6baed6"],
        "sim": ["#d62728", "#fb6a4a"]
    }

    plt.plot(data[0]["size"], data[0]["recall"], marker='o', color=colors["recall"][0], label="Recall (overlap=0)")
    plt.plot(data[50]["size"], data[50]["recall"], marker='o', linestyle='--', color=colors["recall"][1], label="Recall (overlap=50)")

    plt.plot(data[0]["size"], data[0]["sim"], marker='^', color=colors["sim"][0], label="Sim (overlap=0)")
    plt.plot(data[50]["size"], data[50]["sim"], marker='^', linestyle='--', color=colors["sim"][1], label="Sim (overlap=50)")

    plt.xlabel("Chunk Size")
    plt.ylabel("Score")
    plt.title("Novel Chunk Evaluation (Fixed)")
    plt.legend()
    plt.grid()

    plt.savefig("../results/novel_chunk.png")
    plt.show()


if __name__ == "__main__":
    run_all()