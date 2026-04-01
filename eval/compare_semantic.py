import json
import numpy as np
import faiss
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# ===== 配置 =====
CONFIGS = {
    "chunk_300_overlap_50": "../data/index/chunk_300_overlap_50",
    "semantic": "../data/index/semantic"
}


# ===== 相似度 =====
def similarity_vec(a_vec, b_vec):
    return cosine_similarity([a_vec], [b_vec])[0][0]


# ===== 找 GT doc =====
def find_gt_doc(answer_emb, doc_embeddings):
    sims = cosine_similarity([answer_emb], doc_embeddings)[0]
    return int(np.argmax(sims))


# ===== 检索 =====
def retrieve(index, query, top_k=5):
    q_emb = model.encode([query])
    D, I = index.search(q_emb, top_k)
    return I[0]


# ===== RAG回答 =====
def rag_answer(docs, retrieved_ids):
    context = "\n".join([docs[i] for i in retrieved_ids])
    return context[:300]


# ===== 单模型评估 =====
def evaluate(path, dataset, answer_embeddings):

    index = faiss.read_index(f"{path}/knowledge.index")
    docs = np.load(f"{path}/docs.npy", allow_pickle=True)

    doc_embeddings = model.encode(docs, batch_size=64)

    recalls = []
    sims = []

    for i, item in enumerate(dataset):

        q = item["question"]
        gt = item["answer"]
        gt_emb = answer_embeddings[i]

        gt_id = find_gt_doc(gt_emb, doc_embeddings)

        retrieved_ids = retrieve(index, q)

        r = int(gt_id in retrieved_ids)

        pred = rag_answer(docs, retrieved_ids)
        pred_emb = model.encode([pred])[0]

        sim = similarity_vec(pred_emb, gt_emb)

        recalls.append(r)
        sims.append(sim)

    return np.mean(recalls), np.mean(sims)


# ===== 主函数 =====
def run():

    with open("../data/processed/test.json", encoding="utf-8") as f:
        dataset = json.load(f)

    answers = [item["answer"] for item in dataset]
    answer_embeddings = model.encode(answers, batch_size=32)

    names = []
    recalls = []
    sims = []

    for name, path in CONFIGS.items():
        print(f"🚀 Evaluating {name}")

        r, s = evaluate(path, dataset, answer_embeddings)

        names.append(name)
        recalls.append(r)
        sims.append(s)

        print(f"Recall: {r:.3f}, Similarity: {s:.3f}")

    # ===== 画条形图 =====
    x = np.arange(len(names))

    width = 0.35

    plt.figure()

    plt.bar(x - width/2, recalls, width, label="Recall")
    plt.bar(x + width/2, sims, width, label="Similarity")

    plt.xticks(x, names)
    plt.ylabel("Score")
    plt.title("Semantic vs Fixed Chunk Comparison")

    plt.legend()
    plt.grid(axis="y")

    plt.savefig("../results/semantic_vs_chunk.png")
    plt.show()


if __name__ == "__main__":
    run()