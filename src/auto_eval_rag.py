# %% [markdown]
# # 数据第一次清洗文件

# %%
from pathlib import Path
import pickle
import numpy as np
import faiss
import pandas as pd
from dataclasses import dataclass
import time

CACHE_DIR = "D:\WorkDirectory\PythonProject\RAG\cache"
class CacheLoader:
    def __init__(self,cache_dir,task_name:str = None):
        self.cache_dir  = Path(cache_dir)
        # 检查路径
        if not self.cache_dir.exists():
            raise ImportError('文件路径不存在')
        if task_name is None:
            task_name = str(time.time())
        self.task_name = task_name
        self.task_dir = self.cache_dir/self.task_name
        self.task_dir.mkdir(parents=True,exist_ok=True)

        self.chunks_path = self.task_dir/'chunks.pkl'
        self.embeddings_path = self.task_dir/'embeddings.npy'
        self.index_path = self.task_dir/'index.faiss'

        self._chunks =None
        self._embeddings=None
        self._index = None
    '-----存在检查-----'
    def has_chunks(self):
        return self.chunks_path.exists()
    def has_embeddings(self):
        return self.embeddings_path.exists()
    def has_index(self):
        return self.index_path.exists()
    

    @property
    def chunks(self):
        if self._chunks is None and self.chunks_path.exists():
            with open(self.chunks_path,'rb')as f:
                self._chunks = pickle.load(f)
        return self._chunks


    @property
    def embeddings(self):
        if self._embeddings is None and self.embeddings_path.exists():
            self._embeddings = np.load(self.embeddings_path)
        return self._embeddings

    
    @property
    def index(self):
        if self._index is None and self.index_path.exists():
            self._index = faiss.read_index(str(self.index_path))
        return self._index
    
    @chunks.setter
    def chunks(self, value):
        self._chunks = value

    @embeddings.setter
    def embeddings(self, value):
        self._embeddings = value

    @index.setter
    def index(self, value):
        self._index = value
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
loader = CacheLoader(CACHE_DIR,task_name=TASK_NAME)

@dataclass
class CacheConfig:
    root_dir:Path
    task_name:str
    embed_model_name:str
    generate_model_name:str
class CacheManager:
    def __init__(self,config:CacheConfig):
        self.config = config
        self.task_dir = config.root_dir/config.task_name
        self.task_dir.mkdir(parents=True,exist_ok=True)

    def get_path(self,name:str,ext:str = "pkl")->Path:
        return self.task_dir/f'{name}.{ext}'
    def exists(self,name:str,ext:str="pkl")->bool:
        return self.get_path(name,ext).exists()
    def save(self,name:str,data,ext:str="pkl",ser_func=None):
        path = self.get_path(name,ext)
        if ser_func is None:
            with open(path,"wb") as f:
                pickle.dump(data,f)
        else:
            ser_func(path,data)
    def load(self,name:str,ext:str="pkl",deser_func=None):
        path = self.get_path(name,ext)
        if not path.exists():
            return None
        if deser_func is None:
            with open(path,"rb") as f:
                return pickle.load(f)
        else:
            return deser_func(path)
class ModelCache:
    def __init__(self,cache_mgr:CacheManager,model_name:str):
        self.cache_mgr = cache_mgr
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        if self._model is not None:
            return self._model
        model_path = self.cache_mgr.get_path("embedding_model",ext="pt")

        if model_path.exists():
            
            self._model = SentenceTransformer(str(model_path))
        else:
            self._model = SentenceTransformer(self.model_name)
            self._model.save(str(model_path))
        return self._model

# %%
# 读取文件

import pandas as pd 
import pickle
from pathlib import Path
if not loader.has_chunks():
    corpus_df = pd.read_parquet(r"D:\WorkDirectory\PythonProject\RAG\my_dataset\wikidata\data\rag-mini-wikipedia\data\passages.parquet\part.0.parquet")
    # 文档分块
    from functions import FixedSizeChunkStrategy
    chunker  = FixedSizeChunkStrategy(chunk_size=500, overlap=50)
    # all_chunks内部需要展开，不能是列表嵌套列表（需要扁平化）
    all_chunks  = []
    for _,row in corpus_df.iterrows():
        chunks = chunker.chunk(row['passage'])
        all_chunks.extend(chunks)
    loader._chunks = all_chunks
    with open(loader.chunks_path,"wb") as f:
        pickle.dump(all_chunks,f)
    print(all_chunks)


# %%
# 向量嵌入
# 准备一个文本嵌入向量模型
from sentence_transformers import SentenceTransformer
if not loader.has_embeddings():
    all_chunks = loader.chunks 
    # TODO 可以写成使用transformers的复杂做法，需要做池化提取句向量，以及批量编码
    cache_model_dir = Path("./cache/models")
    if cache_model_dir.exists():
        embed_model = SentenceTransformer(str(cache_model_dir),local_files_only=True)
    else:
        embed_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        embed_model.save(str(cache_model_dir))
    # model.save_pretrained(<path>)
    embed_model.save('./cache/models')

    # 将RAG分块嵌入成向量
    # 能直接全部编码，不需要自己实现
    embeddings = embed_model.encode(all_chunks, batch_size=64, convert_to_tensor=False)
    embeddings = embeddings.astype('float32')
    loader.embeddings = embeddings
    np.save(loader.embeddings_path,embeddings)
    # TODO 优化1：调整batchsize
    # TODO 优化2：对chunk按长度预排序
    # TODO 优化3：启用FP16


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


# %%
query_df = pd.read_parquet(r'D:\WorkDirectory\PythonProject\RAG\my_dataset\wikidata\data\rag-mini-wikipedia\data\test.parquet\part.0.parquet')
print(query_df.head(5))
# USE_DATA_COUNT = query_df.shape[0]
USE_DATA_COUNT = 5
query_df = query_df.head(USE_DATA_COUNT)

# %%
questions_list = []
answers_list = []
for _,row in query_df.iterrows():
    question = row['question']
    answer = row['answer']
    questions_list.append(question)
    answers_list.append(answer)

questions_embedding = embed_model.encode(questions_list, batch_size=64, convert_to_tensor=False)
questions_embedding = questions_embedding.astype('float32')
# embeddings = model.encode(answers_list, batch_size=64, convert_to_tensor=False)


# %%
k = 5
distances, indices = index.search(questions_embedding, k)


# %%
print(f"chunks 长度: {len(all_chunks)}")
print(f"索引中向量总数: {index.ntotal}")

# %%
# 假设 chunks 是你分块后的原始文本列表，顺序与 embeddings 相同
# retrieved_chunks = [all_chunks[i] for i in indices[0]]

# %%
# 拼接检索到的内容作为上下文
# context = "\n\n".join(retrieved_chunks)
prompts_list = []
contexts_list = []
for i in range(len(questions_list)):
    retrieved_chunks = [all_chunks[j] for j in indices[i]]
    context = "\n\n".join(retrieved_chunks)
    contexts_list.append(context)
    prompt = f"""基于以下参考信息回答问题。
    参考信息：
    {context}
    只回答问题，不要生成新的问题或者回答问题以外的回答。
    回答结束时，输出eos标志
    问题：{questions_list[i]}
    回答："""
    prompts_list.append(prompt)
prompts_list


# %%


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
torch.cuda.empty_cache()
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # 启用4-bit量化[reference:1]
    bnb_4bit_quant_type="nf4",      # 使用nf4量化类型，质量更高[reference:2]
    bnb_4bit_compute_dtype=torch.float16, # 计算时使用float16，平衡速度与精度[reference:3]
    bnb_4bit_use_double_quant=True, # 启用双重量化，进一步压缩显存[reference:4]
)
# 加载模型和分词器 (以4-bit量化为例)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
generate_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    quantization_config=bnb_config,
    device_map="auto"
)
responses_list = []
for i in range(len(prompts_list)):
    inputs = tokenizer(prompts_list[i],return_tensors="pt").to(generate_model.device)
    response_ids = generate_model.generate(inputs.input_ids,max_new_tokens=32,do_sample=False,stop_strings=["\n"],tokenizer=tokenizer,eos_token_id=tokenizer.eos_token_id)
    input_len = inputs.input_ids.shape[1]
    response_text = tokenizer.decode(response_ids[0][input_len:],skip_special_tokens=True)
    
    print(f"Prompt: {prompts_list[i]}{response_text}")
    responses_list.append(response_text)
    print("-" * 50)



# %%
# 我现在有了参考内容、问题、回答、标准回答的矩阵
# 进行评估
import re
import json
def extract_json(text):
    match = re.search(r'\{.*?"factual_score".*?"completeness_score".*?"redundancy_score".*?\}', text, re.DOTALL)
    print("匹配结果",match)
    if match:
        return json.loads(match.group())
    return None
eval_scores = []
# for i in range(len(prompts_list)):
for i in range(1):
    eval_prompt = f"""你是一个RAG评估专家。请根据标准回答，评估生成回答的质量。

    问题：{questions_list[i]}
    标准回答：{answers_list[i]}
    检索到的上下文：{contexts_list[i]}
    生成回答：{responses_list[i]}

    请从以下维度打分（1-5分）：
    1. 事实一致性：生成回答的信息是否与标准回答一致？（5=完全一致，无矛盾）
    2. 完整性：生成回答是否覆盖了标准回答的所有关键点？（5=完全覆盖）
    3. 冗余性：生成回答是否包含多余或无关信息？（1=严重冗余，5=简洁相关）

    最终输出JSON格式：{{"factual_score": x, "completeness_score": y, "redundancy_score": z}}\n\n
    """
    eval_input = tokenizer(eval_prompt,return_tensors="pt").to(generate_model.device)
    input_len = eval_input.input_ids.shape[0]
    eval_response_ids = generate_model.generate(eval_input.input_ids,max_new_tokens=128,do_sample=False,eos_token_id = tokenizer.eos_token_id)
    eval_response_text = tokenizer.decode(eval_response_ids[0][input_len:],skip_special_tokens=True)
    print("评估结果",eval_response_text)
    try:
        eval_json = extract_json(eval_response_text)
    except Exception as e:
        # print(eval_response_text,"\n",e)
        eval_json = None
    eval_scores.append(eval_json)
eval_scores


