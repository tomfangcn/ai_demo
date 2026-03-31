from pymilvus import MilvusClient,DataType
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils._pytree")

current_dir = Path(__file__).parent
dbpath = current_dir.parent / ".milvus_db" / "my_knowledge.db"


print(dbpath)
client = MilvusClient(dbpath.as_posix())

collection_name = "my_emb_docs"

# 如果集合已存在，先删除（开发环境方便重复测试）
if client.has_collection(collection_name):
    client.drop_collection(collection_name)

docs = [
    "人工智能在1956年被确立为一门学术学科。",
    "艾伦·图灵是第一个在人工智能领域进行深入研究的人。",
    "图灵出生于伦敦的麦达维尔，在英格兰南部长大。"
]

# 使用 BGE-M3 模型（中文优化，本地运行）
bge_m3_ef = BGEM3EmbeddingFunction(
    model_name='BAAI/bge-m3', # Specify the model name
    device='cpu', # Specify the device to use, e.g., 'cpu' or 'cuda:0'
    use_fp16=False, # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
    batch_size=8 # 如果内存或显存有限
)

# 向量化文档
vectors = bge_m3_ef.encode_documents(docs)


# 提取稠密向量（返回的是 dict，包含 'dense' 字段）
dense_vectors = vectors['dense']  # shape: (3, 1024)

dense_dim = len(dense_vectors[0])

# ========== 4. 创建 Schema（表结构） ==========
# 使用 create_schema 方法定义字段
schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True
)

schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dense_dim)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)

# 创建集合
client.create_collection(
    collection_name=collection_name,
    schema=schema
)

# ========== 5. 插入数据 ==========
data = []
for i, (doc, vec) in enumerate(zip(docs, dense_vectors)):
    data.append({
        "id": i,
        "vector": vec.tolist(),
        "text": doc
    })

insert_res = client.insert(collection_name=collection_name, data=data)
print(f"插入 {insert_res['insert_count']} 条数据")

# ========== 6. 创建索引（加速搜索） ==========
# 准备索引参数
index_params = MilvusClient.prepare_index_params()

index_params.add_index(
    field_name="vector",          # 要创建索引的向量字段
    index_type="IVF_FLAT",        # 索引类型，适合中等规模数据
    metric_type="COSINE",         # 相似度度量：余弦相似度
    params={"nlist": 128},        # 聚类中心数量
    index_name="vector_idx"
)

# 创建索引
client.create_index(
    collection_name=collection_name,
    index_params=index_params
)

print("索引创建成功")

# ========== 7. 验证：执行搜索 ==========
# 搜索与第一条文档最相似的内容
query_vector = dense_vectors[0].tolist()  # 用第一条文档的向量作为查询

search_res = client.search(
    collection_name=collection_name,
    data=[query_vector],
    limit=3,
    output_fields=["text"]  # 返回 text 字段
)

print("\n搜索结果：")
for hits in search_res:
    for hit in hits:
        print(f"  ID: {hit['id']}, 相似度: {hit['distance']:.4f}")
        print(f"  文本: {hit['entity']['text']}")
