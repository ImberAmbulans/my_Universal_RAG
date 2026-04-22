
from pathlib import Path
import pickle
import numpy as np
import faiss
import pandas as pd
from dataclasses import dataclass
import time
from sentence_transformers import SentenceTransformer 


@dataclass
class CacheConfig:
    root_dir:Path
    task_name:str
class CacheManager:
    def __init__(self,config:CacheConfig,dir_name:str = 'cache'):
        self.config = config
        self.task_dir = config.root_dir/dir_name/config.task_name
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