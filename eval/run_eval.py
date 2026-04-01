import json
import numpy as np
import faiss
import os
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ===== 1. 模型 =====
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# ===== 2. 所有实验配置 =====
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


BASE_PATH = "../data/index"


# ===== 3. 相似度 =====
def similarity_vec(a_vec, b_vec):
    return cosine_similarity([a_vec], [b_vec])[0][0]


# ===== 4. 找 GT doc（语义版🔥）=====
def find_gt_doc(answer_emb, doc_embeddings):
    sims = cosine_similarity([answer_emb], doc_embeddings)[0]
    return int(np.argmax(sims))


# ===== 5. 检索 =====
def retrieve(index, query, top_k=5):
    q_emb = embed_model.encode([query])
    D, I = index.search(q_emb, top_k)
    return I[0]


# ===== 6. RAG回答 =====
def rag_answer(docs, retrieved_ids):
    context = "\n".join([docs[i] for i in retrieved_ids])
    return context[:300]


# ===== 7. 单个配置评估 =====
def evaluate_config(config_name, dataset, answer_embeddings):

    print(f"\n🚀 Running config: {config_name}")

    index_path = f"{BASE_PATH}/{config_name}/knowledge.index"
    docs_path = f"{BASE_PATH}/{config_name}/docs.npy"

    index = faiss.read_index(index_path)
    docs = np.load(docs_path, allow_pickle=True)

    # 预计算doc embedding
    doc_embeddings = embed_model.encode(docs, batch_size=64, show_progress_bar=False)

    recalls = []
    sims = []

    for idx, item in enumerate(dataset):

        q = item["question"]
        gt = item["answer"]
        gt_emb = answer_embeddings[idx]

        # GT doc
        gt_id = find_gt_doc(gt_emb, doc_embeddings)

        # 检索
        retrieved_ids = retrieve(index, q, top_k=5)

        # Recall
        r = int(gt_id in retrieved_ids)

        # 生成
        pred = rag_answer(docs, retrieved_ids)

        pred_emb = embed_model.encode([pred])[0]
        sim = similarity_vec(pred_emb, gt_emb)

        recalls.append(r)
        sims.append(sim)

    return {
        "recall": np.mean(recalls),
        "similarity": np.mean(sims)
    }


# ===== 8. 主函数 =====
def run_all():

    with open("../data/processed/test.json", encoding="utf-8") as f:
        dataset = json.load(f)

    # 预计算答案embedding
    answers = [item["answer"] for item in dataset]
    answer_embeddings = embed_model.encode(answers, batch_size=32)

    results = []

    for config in CONFIGS:
        res = evaluate_config(config, dataset, answer_embeddings)
        res["config"] = config
        results.append(res)

    # ===== 打印结果 =====
    print("\n===== ALL RESULTS =====")
    for r in results:
        print(r)

    # ===== 可视化 =====
    plot_results(results)


# ===== 9. 可视化 =====
def plot_results(results):

    import matplotlib.pyplot as plt
    import numpy as np

    # ===== 按 overlap 分组 =====
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
        data[overlap]["sim"].append(r["similarity"])

    # ===== 排序 =====
    for overlap in data:
        idx = np.argsort(data[overlap]["size"])

        for key in ["size", "recall", "sim"]:
            data[overlap][key] = np.array(data[overlap][key])[idx]

    # ===== 画图 =====
    plt.figure()

    # 🎨 配色
    colors = {
        "recall": ["#1f77b4", "#6baed6"],   # 蓝
        "sim": ["#d62728", "#fb6a4a"]        # 红
    }

    # ===== Recall =====
    plt.plot(
        data[0]["size"], data[0]["recall"],
        marker='o', color=colors["recall"][0],
        label="Recall (overlap=0)"
    )
    plt.plot(
        data[50]["size"], data[50]["recall"],
        marker='o', linestyle='--', color=colors["recall"][1],
        label="Recall (overlap=50)"
    )

    # ===== Similarity =====
    plt.plot(
        data[0]["size"], data[0]["sim"],
        marker='^', color=colors["sim"][0],
        label="Similarity (overlap=0)"
    )
    plt.plot(
        data[50]["size"], data[50]["sim"],
        marker='^', linestyle='--', color=colors["sim"][1],
        label="Similarity (overlap=50)"
    )

    # ===== 美化 =====
    plt.xlabel("Chunk Size")
    plt.ylabel("Score")
    plt.title("RAG Performance vs Chunk Size (with Overlap Comparison)")

    plt.legend()
    plt.grid()

    plt.savefig("../results/chunk_overlap_experiment.png")
    plt.show()


if __name__ == "__main__":
    run_all()