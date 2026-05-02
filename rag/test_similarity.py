from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

s1 = "从那时起镇上的事便由她做主"
s2 = "她恢复星期天的弥撒，停用红袖章，废除那些轻率无理的条令"

# 计算相似度
emb1 = model.encode(s1).reshape(1, -1)
emb2 = model.encode(s2).reshape(1, -1)
sim = cosine_similarity(emb1, emb2)[0][0]

print("句子1：", s1)
print("句子2：", s2)
print(f"余弦相似度：{sim:.4f}")