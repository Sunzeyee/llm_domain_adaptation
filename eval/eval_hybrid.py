import json
import numpy as np
import faiss
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import jieba


# ===== 1. 模型 =====
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# ===== 2. 路径 =====
INDEX_PATH = "../data/index/chunk_300_overlap_50/knowledge.index"
DOCS_PATH = "../data/index/chunk_300_overlap_50/docs.npy"


# ===== 3. 加载 =====
index = faiss.read_index(INDEX_PATH)
docs = np.load(DOCS_PATH, allow_pickle=True)


# ===== 4. 分词 =====
def tokenize(text):
    return list(jieba.cut(text))


# ===== 5. BM25 =====
tokenized_docs = [tokenize(doc) for doc in docs]
bm25 = BM25Okapi(tokenized_docs)


# ===== 6. 预计算 doc embedding =====
doc_embeddings = embed_model.encode(docs, batch_size=64, show_progress_bar=True)


# ===== 7. 相似度 =====
def similarity_vec(a_vec, b_vec):
    return cosine_similarity([a_vec], [b_vec])[0][0]


# ===== 8. 找 GT doc =====
def find_gt_doc(answer_emb):
    sims = cosine_similarity([answer_emb], doc_embeddings)[0]
    return int(np.argmax(sims))

def normalize(scores):
    min_v = np.min(scores)
    max_v = np.max(scores)
    if max_v - min_v < 1e-6:
        return np.zeros_like(scores)
    return (scores - min_v) / (max_v - min_v)

# ===== 9. Hybrid 检索 =====
def hybrid_retrieve(query, alpha, top_k=5, dense_k=100):

    q_emb = embed_model.encode([query])

    # ===== Dense TopK（关键🔥）=====
    D, I = index.search(q_emb, dense_k)

    dense_ids = I[0]
    dense_scores = -D[0]

    # 只在 dense候选集上做融合
    dense_scores = normalize(dense_scores)

    # ===== BM25（只算候选集）=====
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

# ===== 10. RAG =====
def rag_answer(retrieved_ids):
    context = "\n".join([docs[i] for i in retrieved_ids])
    return context[:300]


# ===== 11. 单个alpha评估 =====
def evaluate_alpha(alpha, dataset, answer_embeddings):

    recalls = []
    sims = []

    for idx, item in enumerate(dataset):

        q = item["question"]
        gt_emb = answer_embeddings[idx]

        gt_id = find_gt_doc(gt_emb)

        retrieved_ids = hybrid_retrieve(q, alpha)

        r = int(gt_id in retrieved_ids)

        pred = rag_answer(retrieved_ids)
        pred_emb = embed_model.encode([pred])[0]
        sim = similarity_vec(pred_emb, gt_emb)

        recalls.append(r)
        sims.append(sim)

    return np.mean(recalls), np.mean(sims)


# ===== 12. 主函数 =====
def run_all():

    with open("../data/processed/test.json", encoding="utf-8") as f:
        dataset = json.load(f)

    answers = [item["answer"] for item in dataset]
    answer_embeddings = embed_model.encode(answers, batch_size=32)

    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    recall_list = []
    sim_list = []

    for alpha in alphas:
        print(f"\n🚀 Alpha = {alpha}")

        recall, sim = evaluate_alpha(alpha, dataset, answer_embeddings)

        recall_list.append(recall)
        sim_list.append(sim)

        print(f"Recall={recall:.3f}, Sim={sim:.3f}")

    plot_curve(alphas, recall_list, sim_list)


# ===== 13. 画图 =====
def plot_curve(alphas, recall_list, sim_list):

    plt.figure()

    plt.plot(alphas, recall_list, marker='o', label="Recall")
    plt.plot(alphas, sim_list, marker='^', label="Similarity")

    plt.xlabel("Alpha (BM25 Weight)")
    plt.ylabel("Score")
    plt.title("Hybrid Retrieval Performance vs Alpha")

    plt.legend()
    plt.grid()

    plt.savefig("../results/alpha_experiment.png")
    plt.show()


if __name__ == "__main__":
    run_all()