# 自动化流程、评估、升级

#%%

from datasets import load_dataset
from pathlib import Path


ROOT_DIR = Path(r'D:\WorkDirectory\PythonProject\RAG\\')
CACHE_DIR = ROOT_DIR/"my_dataset"
datasets = []
dataset = load_dataset("imdb",cache_dir=CACHE_DIR)
datasets.append(dataset)
# dataset = load_dataset("Youtu-Graph/AnonyRAG", "AnnoyRAG-CHS-QA",cache_dir=CACHE_DIR)
# datasets.append(dataset)
# dataset = load_dataset("Youtu-Graph/AnonyRAG", "AnnoyRAG-CHS-Texts",cache_dir=CACHE_DIR)
# datasets.append(dataset)

