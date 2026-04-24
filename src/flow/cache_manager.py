
from pathlib import Path
import pickle
import numpy as np
import faiss
import pandas as pd
from dataclasses import dataclass
import time
from sentence_transformers import SentenceTransformer 
import torch

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
    def _get_serializer(self, ext: str):
        """返回 (save_func, load_func) 对应扩展名"""
        if ext == 'pkl':
            return (
                lambda p, d: pickle.dump(d, open(p, 'wb')),
                lambda p: pickle.load(open(p, 'rb'))
            )
        elif ext == 'npy':
            return (
                lambda p, d: np.save(p, d),
                lambda p: np.load(p, allow_pickle=True)  # allow_pickle 根据需要
            )
        # elif ext == 'pt':
        #     return (
        #         lambda p, d: torch.save(d, p),
        #         lambda p: torch.load(p, map_location='cpu')
        #     )
        elif ext == 'faiss':
            def save_faiss(p, d):
                faiss.write_index(d, str(p))
            def load_faiss(p):
                return faiss.read_index(str(p))
            return (save_faiss, load_faiss)
        else:
            raise ValueError(f"Unsupported extension: {ext}")
    def save(self,name:str,data,ext:str="pkl",ser_func=None):
        path = self.get_path(name,ext)
        if ser_func is not None:
            ser_func(path,data)
        else:
            save_func,_ = self._get_serializer(ext)
            save_func(path,data)
    def load(self,name:str,ext:str="pkl",deser_func=None):
        path = self.get_path(name,ext)
        if not path.exists():
            return None
        if deser_func is not None:
            return deser_func(path)
        else:
            _,load_func = self._get_serializer(ext)
            return load_func(path)