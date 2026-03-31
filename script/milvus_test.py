from pymilvus import MilvusClient
import numpy as np
from pathlib import Path

current_dir = Path(__file__).parent
dbpath = current_dir.parent / ".milvus_db" / "my_knowledge.db"

print(dbpath)
client = MilvusClient(dbpath.as_posix())

client.create_collection(
    collection_name="my_docs",
    dimension=384  # 这里假设你的 embedding 模型是384维
)

docs = [
    "人工智能在1956年被确立为一门学术学科。",
    "艾伦·图灵是第一个在人工智能领域进行深入研究的人。",
    "图灵出生于伦敦的麦达维尔，在英格兰南部长大。"
]


vectors = [[np.random.uniform(-1, 1) for _ in range(384)] for _ in range(len(docs))]
data = [{"id": i, "vector": vectors[i], "text": docs[i]} for i in range(len(docs))]
client.insert(collection_name="my_docs", data=data)


# 搜索与第一个向量最相似的文档
res = client.search(
    collection_name="my_docs",
    data=[vectors[0]],
    limit=2,
    output_fields=["text"]
)
print(res)

