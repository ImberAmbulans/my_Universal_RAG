# RAG/src/main.py



from pathlib import Path
from dataset import Dataset,DatasetManager

from config import ROOT_DIR,CACHE_PATH,DATASET_PATH

dsmgr = DatasetManager(base_path=DATASET_PATH)
print(dsmgr.list())

input()
ds1 = Dataset(
    name="Congliu/Chinese-DeepSeek-R1-Distill-data-110k",
    displayname="HF_Distill_110K",
    platform="huggingface",
    remote_path="Congliu/Chinese-DeepSeek-R1-Distill-data-110k",
    base_path=str(DATASET_PATH),
    local_path="Congliu/Chinese-DeepSeek-R1-Distill-data-110k",
    enabled=True
)


datasets = [
    ds1
]

dsmgr.add(ds1,True)

print(dsmgr.list())

# dataset = load_dataset("imdb",cache_dir=CACHE_DIR)
# datasets.append(dataset)
# dataset = load_dataset("Youtu-Graph/AnonyRAG", "AnnoyRAG-CHS-QA",cache_dir=CACHE_DIR)
# datasets.append(dataset)
# dataset = load_dataset("Youtu-Graph/AnonyRAG", "AnnoyRAG-CHS-Texts",cache_dir=CACHE_DIR)
# datasets.append(dataset)

