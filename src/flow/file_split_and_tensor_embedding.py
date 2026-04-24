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
        TASK_NAME = parts[idx + 2]
    else:
        TASK_NAME = "unknown"
except ValueError:
    TASK_NAME = "unknown"
print(TASK_NAME)

config = CacheConfig(root_dir,TASK_NAME)
# mgr = CacheManager(config=config)


def cache_check(results:list[tuple[str,str]],sources:list[tuple[str,str]],mgr:CacheManager=None):
    if mgr is None:
        default_config = CacheConfig(root_dir=Path.cwd(), task_name="default")
        mgr = CacheManager(default_config)
    def decorater(func):
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
def chunking():
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



@cache_check(
        results=[
            ('embeddings','npy')
        ],
        sources=[
            ('all_chunks','pkl')
        ]
)
def embedding(**kwargs):
    all_chunks = kwargs['all_chunks']
    MODEL_NAME = 'Qwen/Qwen3-Embedding-0.6B'
    model_cache_dir =root_dir/Path(f"./cache/models/{MODEL_NAME}")
    if (model_cache_dir/'sentence_transformers.json').exists():
        embed_model = SentenceTransformer(str(model_cache_dir),local_files_only=True)
    else:
        embed_model = SentenceTransformer(f"{MODEL_NAME}")
        embed_model.save_pretrained(str(model_cache_dir))
    
    embeddings = embed_model.encode(all_chunks, batch_size=64, convert_to_tensor=False)
    embeddings = embeddings.astype('float32')
    return {
        'embeddings':embeddings
    }


import faiss
import numpy as np
# 需求：embedding (float32类型数组)

@cache_check(
        results=[
            ('index','faiss')
        ],
        sources=[
            ('embeddings','npy')
        ]
)
def indexing(**kwargs):
    embeddings = kwargs['embeddings']
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension) 
    index.add(embeddings)
    return {'index':index}


if __name__ == "__main__":
    chunking()
    embedding()
    indexing()