from pathlib import Path
from dataclasses import dataclass
from typing import List
import json
# 我想实现的是以下的功能
# 添加展示名字displayname,
# 添加是否使用这个数据集标签 isuse/tag, 使用时属性，应该写在数据集配置中吗？
# 另外的脚本，增删改查这个列表的功能
# 对数据集的来源进行提示的功能，影响读取/下载/加载的方式
# 兼容？比如imdb的数据集能用test/split/定界符切片，但是其他来源的数据集可能本身做不到这一点。
# 添加数据集的时候，是否要尝试读取内容进行属性的确定？


# 获取本地数据集存储地址
from config import DATASET_PATH


@dataclass
class Dataset:
    name:str
    displayname:str
    platform:str
    remote_path:str
    local_path:str
    enabled:bool = True

    @property
    def full_local_path(self) -> Path:
        return DATASET_PATH / self.local_path

class DatasetDownloader:






    def __init__(self,path:Path):
        self.path = DATASET_PATH
        self.path.mkdir(parents=True,exist_ok=True)

class DatasetLoader:



    def __init__(self,path:Path):
        self.path = DATASET_PATH
    
    def __enter__(self):
        pass

    def __exit__(self):
        pass



# 单例类
class DatasetManager:
    _instance = None
    _storage_file = DATASET_PATH / "datasets.json"
    datasets:List[Dataset]
    @staticmethod
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.datasets = []
            cls._instance._load()
        return cls._instance


    # 持久化
    def _load(self):
        """从 JSON 文件加载数据集列表"""
        if self._storage_file.exists():
            with open(self._storage_file, "r", encoding="utf-8") as f:
                raw_list = json.load(f)
            self.datasets = [Dataset(**item) for item in raw_list]
        else:
            self.datasets = []

    def _save(self):
        """将当前数据集列表写回 JSON"""
        with open(self._storage_file, "w", encoding="utf-8") as f:
            json.dump([d.__dict__ for d in self.datasets], f, indent=2, ensure_ascii=False)

    # CRUD
    def add(self,dataset:Dataset):
        if any(d.name ==dataset.name for d in self.datasets):
            raise ValueError(f"数据集{dataset.name}已存在")
        self.datasets.append(dataset)
        self._save()
    
    def get(self,name:str):
        for d in self.datasets:
            if d.name == name:
                return d
        return None
    
    def remove(self,name:str):
        ds = self.get(name)
        if ds is None:
            raise KeyError(f'数据集{name}不存在')
        self.datasets.remove(ds)
        self._save()

    def update(self,name:str,**kwargs):
        ds = self.get(name)
        if ds is None:
            raise KeyError(f"数据集{name}不存在")
        # TODO 更新逻辑存在一定问题
        for key,value in kwargs.items():
            if hasattr(ds,key):
                setattr(ds,key,value)
        self._save()




