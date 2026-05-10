from datasets import load_dataset
from pathlib import Path


ROOT_DIR = Path(r'D:\WorkDirectory\PythonProject\RAG\\')
CACHE_DIR = ROOT_DIR/"my_dataset"
dataset = load_dataset("imdb",cache_dir=CACHE_DIR) 

# # 加载名为 "imdb" 的经典数据集
# dataset = load_dataset("imdb",cache_dir=CACHE_DIR) 
# # 加载特定命名空间下的数据集
# data = load_dataset("jeffnyman/emotions") 

# # 加载本地的 CSV 文件
# dataset = load_dataset('csv', data_files='./data/dataset.csv')
# # 同时加载本地的训练集和测试集 CSV 文件
# dataset_dict = load_dataset('csv', data_files={'train': './data/train.csv', 'test': './data/test.csv'})

# 除了 load_dataset，Hugging Face datasets 库还提供了两个功能更聚焦的本地数据加载函数。

# load_from_disk：高效加载本地二进制数据。
# 它用于加载通过 save_to_disk 方法保存的、经优化的二进制格式数据集。这种方式加载速度更快、占用存储空间更小，并能完整保留数据集的特征和元数据。

# from datasets import load_from_disk
# dataset = load_from_disk("dataset_path")
# 当数据集规模庞大（达到 TB 级别）时，load_dataset 的 streaming=True 参数变得非常有价值。它允许以流式方式加载数据，即边下载边处理，无需等待完整下载，能极大节省磁盘空间，非常适合大规模训练的场景。

# from datasets import load_dataset
# # 以流式方式加载，无需等待完整下载
# dataset = load_dataset("HuggingFaceM4/FineVisionMax", split="train", streaming=True)
# # 获取第一个样本开始处理
# print(next(iter(dataset))) 


ds = load_dataset("Youtu-Graph/AnonyRAG", "AnnoyRAG-CHS-QA",cache_dir=CACHE_DIR)
ds = load_dataset("Youtu-Graph/AnonyRAG", "AnnoyRAG-CHS-Texts",cache_dir=CACHE_DIR)