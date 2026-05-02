import fitz
import numpy as np
import faiss
import os

from sentence_transformers import SentenceTransformer


PDF_PATH = "../data/raw/mianzhanixi.pdf"

# ===== 实验参数（你改这里🔥）=====
CHUNK_SIZE = 100
OVERLAP = 0

SAVE_DIR = f"../data/index/chunk_{CHUNK_SIZE}_overlap_{OVERLAP}"
os.makedirs(SAVE_DIR, exist_ok=True)


# ===== 1. 读取PDF =====
def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# ===== 2. 清洗 =====
def clean_text(text):
    text = text.replace("\r", "")
    return text


# ===== 3. 固定切块🔥 =====
def split_chunks(text, chunk_size, overlap):

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# ===== 4. 构建索引 =====
def build_index():

    print("📄 Loading PDF...")
    text = extract_text(PDF_PATH)

    print("🧹 Cleaning...")
    text = clean_text(text)

    print("✂️ Splitting...")
    docs = split_chunks(text, CHUNK_SIZE, OVERLAP)

    print(f"Chunks: {len(docs)}")

    print("🧠 Embedding...")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(docs, batch_size=64, show_progress_bar=True)

    dim = embeddings.shape[1]

    print("📦 Building FAISS...")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print("💾 Saving...")
    faiss.write_index(index, f"{SAVE_DIR}/knowledge.index")
    np.save(f"{SAVE_DIR}/docs.npy", docs)

    print("✅ Done!")


if __name__ == "__main__":
    build_index()