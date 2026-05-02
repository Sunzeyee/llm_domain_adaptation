import os
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ===== 路径 =====
TXT_PATH = "../data/raw/bainiangudu.txt"
SAVE_DIR = "../data/index/novel_chunk"

CHUNK_SIZES = [100, 200, 300, 400, 500]
OVERLAPS = [0, 50]

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# ===== 1. 读取 =====
def load_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


# ===== 2. 清洗 =====
def clean_text(text):
    text = text.replace("\r", "")
    text = re.sub(r"\n+", "\n", text)
    return text


# ===== 3. 固定分块 =====
def chunk_text(text, chunk_size, overlap):

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start = end - overlap  # 🔥 overlap关键

    return chunks


# ===== 4. 构建 =====
def build_all():

    text = clean_text(load_text(TXT_PATH))

    for size in CHUNK_SIZES:
        for overlap in OVERLAPS:

            name = f"chunk_{size}_overlap_{overlap}"
            save_path = f"{SAVE_DIR}/{name}"
            os.makedirs(save_path, exist_ok=True)

            print(f"\n🚀 Building {name}")

            docs = chunk_text(text, size, overlap)

            embeddings = model.encode(docs, batch_size=64, show_progress_bar=True)

            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)

            faiss.write_index(index, f"{save_path}/knowledge.index")
            np.save(f"{save_path}/docs.npy", docs)

            print(f"Chunks: {len(docs)}")


if __name__ == "__main__":
    build_all()