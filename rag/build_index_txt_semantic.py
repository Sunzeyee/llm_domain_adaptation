import os
import re
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ===== 路径 =====
TXT_PATH = "../data/raw/bainiangudu.txt"
SAVE_DIR = "../data/index/novel_semantic_bge"

THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8]
WINDOW_SIZE = 3   # 🔥关键（小说一定要window）

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")


# ===== 1. 读取 =====
def load_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


# ===== 2. 清洗 =====
def clean_text(text):
    text = text.replace("\r", "")
    text = re.sub(r"\n+", "\n", text)
    return text


# ===== 3. 切句 =====
def split_sentences(text):
    sentences = re.split(r"[。！？]", text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]


# ===== 4. 语义window分块🔥 =====
def semantic_chunk(sentences, threshold):

    embeddings = model.encode(sentences, batch_size=64)

    chunks = []
    current_chunk = sentences[0]

    for i in range(1, len(sentences)):

        # ===== window上下文 =====
        left = max(0, i - WINDOW_SIZE)
        prev_vec = np.mean(embeddings[left:i], axis=0)

        sim = cosine_similarity([prev_vec], [embeddings[i]])[0][0]

        if sim < threshold:
            chunks.append(current_chunk)
            current_chunk = sentences[i]
        else:
            current_chunk += "。" + sentences[i]

    chunks.append(current_chunk)
    return chunks


# ===== 5. 构建 =====
def build_all():

    text = clean_text(load_text(TXT_PATH))
    sentences = split_sentences(text)

    for t in THRESHOLDS:

        name = f"semantic_t{t}"
        save_path = f"{SAVE_DIR}/{name}"
        os.makedirs(save_path, exist_ok=True)

        print(f"\n🚀 Building {name}")

        docs = semantic_chunk(sentences, t)

        embeddings = model.encode(docs, batch_size=64, show_progress_bar=True)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        faiss.write_index(index, f"{save_path}/knowledge.index")
        np.save(f"{save_path}/docs.npy", docs)

        print(f"Chunks: {len(docs)}")


if __name__ == "__main__":
    build_all()