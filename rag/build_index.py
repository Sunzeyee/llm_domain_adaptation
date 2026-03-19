# rag/build_index.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

os.makedirs("../data", exist_ok=True)

# 读取知识库
with open("../data/knowledge.txt", encoding="utf-8") as f:
    docs = f.read().split("\n\n")

# 中文可用多语言模型
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

embeddings = embed_model.encode(docs, convert_to_numpy=True)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 保存索引和原文
faiss.write_index(index, "knowledge.index")
np.save("docs.npy", docs)

print("✅ Knowledge index built!")