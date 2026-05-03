import fitz
import numpy as np
import faiss
import os
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


PDF_PATH = "../data/raw/mianzhanixi.pdf"

# ===== 实验参数 =====
THRESHOLDS = [0.6, 0.7, 0.75, 0.8, 0.85]
OVERLAPS = [0, 50]

BASE_DIR = "../data/index"

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# ===== 1. 读取 PDF =====
def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# ===== 2. 清洗 =====
def clean_text(text):
    text = text.replace("\r", "")
    text = re.sub(r"\n+", "\n", text)

    # 修复英文断行
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    return text


# ===== 3. 分句 =====
def split_sentences(text):
    sentences = re.split(r"[。！？\n]", text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]


# ===== 4. 构建窗口 =====
def build_windows(sentences, window_size=3):
    windows = []

    for i in range(len(sentences)):
        window = sentences[i:i + window_size]
        if len(window) < window_size:
            break
        windows.append(" ".join(window))

    return windows


# ===== 5. 窗口语义分块🔥 =====
def semantic_chunk_window(text, threshold=0.7, window_size=3):

    sentences = split_sentences(text)

    if len(sentences) <= window_size:
        return [" ".join(sentences)]

    windows = build_windows(sentences, window_size)

    embeddings = model.encode(windows, batch_size=64)

    chunks = []
    current_chunk = windows[0]

    for i in range(1, len(windows)):

        sim = cosine_similarity(
            [embeddings[i - 1]],
            [embeddings[i]]
        )[0][0]

        if sim < threshold:
            chunks.append(current_chunk)
            current_chunk = windows[i]
        else:
            current_chunk += " " + windows[i]

    chunks.append(current_chunk)

    return chunks


# ===== 6. 加 overlap（chunk级🔥）=====
def add_overlap(chunks, overlap_size=50):

    if overlap_size == 0:
        return chunks

    new_chunks = []

    for i, chunk in enumerate(chunks):
        if i == 0:
            new_chunks.append(chunk)
        else:
            prev = new_chunks[-1]

            # 👉 用前一个chunk的尾部拼接
            overlap_text = prev[-overlap_size:]

            new_chunks.append(overlap_text + " " + chunk)

    return new_chunks


# ===== 7. 构建索引 =====
def build_all():

    print("📄 Loading PDF...")
    text = extract_text(PDF_PATH)

    print("🧹 Cleaning text...")
    text = clean_text(text)

    for t in THRESHOLDS:
        for o in OVERLAPS:

            name = f"semantic_window_t{t}_overlap_{o}"
            save_dir = f"{BASE_DIR}/{name}"
            os.makedirs(save_dir, exist_ok=True)

            print(f"\n🚀 Building {name}")

            # ===== 核心：窗口语义分块 =====
            chunks = semantic_chunk_window(
                text,
                threshold=t,
                window_size=3
            )

            # ===== 加 overlap =====
            chunks = add_overlap(chunks, overlap_size=o)

            print(f"Chunks: {len(chunks)}")

            # ===== embedding =====
            embeddings = model.encode(chunks, batch_size=64)

            # ===== FAISS =====
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)

            # ===== 保存 =====
            faiss.write_index(index, f"{save_dir}/knowledge.index")
            np.save(f"{save_dir}/docs.npy", chunks)

    print("\n✅ ALL DONE!")


if __name__ == "__main__":
    build_all()