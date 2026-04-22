import pandas as pd 
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from cache_manager import CacheConfig,CacheManager

root_dir = Path.cwd()
root_dir = Path(r'D:\WorkDirectory\PythonProject\RAG\\')

TARGET_FILE = Path(r"D:\WorkDirectory\PythonProject\RAG\my_dataset\wikidata\data\rag-mini-wikipedia\data\passages.parquet\part.0.parquet")
TARGET_FILES = None
parts = TARGET_FILE.parts
try:
    idx = parts.index('RAG')
    # 检查是否有足够的深度
    if len(parts) > idx + 2:
        TASK_NAME = parts[idx + 3]
    else:
        TASK_NAME = "unknown"
except ValueError:
    TASK_NAME = "unknown"
print(TASK_NAME)

config = CacheConfig(root_dir,TASK_NAME)
mgr = CacheManager(config=config)


def cache_check(results:list[tuple[str,str]],sources:list[tuple[str,str]]):
    def decorater(func):
        global mgr
        def wrapper():
            if all(mgr.exists(name,ext) for name,ext in results):
                return None
            source_data  = {}
            for name,ext in sources:
                data = mgr.load(name,ext)
                source_data[name]=data
            generated_data = func(**source_data)
            for (name, ext), (result_name, result_data) in zip(results, generated_data.items()):
                if name != result_name:
                    raise ValueError(f"Result name mismatch: expected {name}, got {result_name}")
                mgr.save(name, result_data, ext)
        return wrapper
    return decorater


@cache_check(
        results=[('all_chunks','pkl')],
        sources=[]
    )
def split_chunks():
    corpus_df = pd.read_parquet(TARGET_FILE)
    # 文档分块
    import sys
    global root_dir
    sys.path.insert(0, str(root_dir))
    from src.functions import FixedSizeChunkStrategy
    chunker  = FixedSizeChunkStrategy(chunk_size=500, overlap=50)
    all_chunks  = []
    for _,row in corpus_df.iterrows():
        chunks = chunker.chunk(row['passage'])
        all_chunks.extend(chunks)
    return {'all_chunks':all_chunks}

split_chunks()
print('到此正确结束')
exit()
# %%
if not loader.has_embeddings():
    all_chunks = loader.chunks 
    # TODO 可以写成使用transformers的复杂做法，需要做池化提取句向量，以及批量编码
    cache_model_dir = Path("./cache/models")
    if cache_model_dir.exists():
        embed_model = SentenceTransformer(str(cache_model_dir),local_files_only=True)
    else:
        embed_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        embed_model.save(str(cache_model_dir))
    embed_model.save('./cache/models')
    embeddings = embed_model.encode(all_chunks, batch_size=64, convert_to_tensor=False)
    embeddings = embeddings.astype('float32')
    loader.embeddings = embeddings
    np.save(loader.embeddings_path,embeddings)


# %%
import faiss
import numpy as np
# 需求：embedding (float32类型数组)
if not loader.has_index():
    embeddings = loader.embeddings
    dimension = embeddings.shape[1]
    # --- 方案 A: 暴力检索 (高精度，小数据量首选) ---
    index = faiss.IndexFlatL2(dimension)  # L2 欧氏距离
    print(f"索引是否需要训练: {index.is_trained}") # Flat索引不需要训练

    # --- 方案 B: 快速近似检索 (大数据量，需要训练) ---
    # nlist = 100  # 聚类中心数量，通常设置为 sqrt(向量数量)
    # quantizer = faiss.IndexFlatL2(dimension)
    # index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    # # 注意: IVF索引在使用前必须经过训练
    # index.train(embeddings)

    # 3. 将向量添加到索引中
    index.add(embeddings)
    faiss.write_index(index,str(loader.index_path))
    print(f"索引中向量总数: {index.ntotal}") # 应等于 embeddings 数量

