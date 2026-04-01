import fitz
import numpy as np
import faiss
import os
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from semantic_chunk_window import semantic_chunk_window

PDF_PATH = "../data/raw/mianzhanixi.pdf"

# ===== 实验参数🔥 =====
THRESHOLDS = [0.6, 0.7, 0.75, 0.8, 0.85]
OVERLAPS = [0, 50]

BASE_DIR = "../data/index"

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# ===== 1. 读取 =====
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
    return text


# ===== 3. 句子 =====
def split_sentences(text):
    sentences = re.split(r"[。！？]", text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]


# ===== 4. 语义分块 + overlap🔥 =====
def semantic_chunk(sentences, threshold, overlap=0):

    embeddings = model.encode(sentences, batch_size=64)

    chunks = []
    current_chunk = sentences[0]

    for i in range(1, len(sentences)):

        sim = cosine_similarity(
            [embeddings[i - 1]],
            [embeddings[i]]
        )[0][0]

        if sim < threshold:
            chunks.append(current_chunk)

            # overlap：回退几句
            if overlap > 0 and len(chunks) > 0:
                overlap_text = chunks[-1][-overlap:]
                current_chunk = overlap_text + sentences[i]
            else:
                current_chunk = sentences[i]
        else:
            current_chunk += "。" + sentences[i]

    chunks.append(current_chunk)
    return chunks


# ===== 5. 构建索引 =====
def build_all():

    text = clean_text(extract_text(PDF_PATH))
    sentences = split_sentences(text)

    for t in THRESHOLDS:
        for o in OVERLAPS:

            name = f"semantic_t{t}_overlap_{o}"
            save_dir = f"{BASE_DIR}/{name}"
            os.makedirs(save_dir, exist_ok=True)

            print(f"\n🚀 Building {name}")

            docs = semantic_chunk(sentences, t, o)

            embeddings = model.encode(docs, batch_size=64)

            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)

            faiss.write_index(index, f"{save_dir}/knowledge.index")
            np.save(f"{save_dir}/docs.npy", docs)

            print(f"Chunks: {len(docs)}")


if __name__ == "__main__":
    build_all()