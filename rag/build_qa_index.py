import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_PATH = "../data/processed/test.json"
SAVE_DIR = "../data/index/qa_index"

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def build():
    os.makedirs(SAVE_DIR, exist_ok=True)

    with open(DATA_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    questions = [item["question"] for item in dataset]
    answers = [item["answer"] for item in dataset]

    print("Embedding questions...")
    q_embeddings = model.encode(questions, batch_size=64)

    dim = q_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(q_embeddings)

    # 保存
    faiss.write_index(index, f"{SAVE_DIR}/qa.index")
    np.save(f"{SAVE_DIR}/questions.npy", questions)
    np.save(f"{SAVE_DIR}/answers.npy", answers)

    print("✅ QA index built!")


if __name__ == "__main__":
    build()