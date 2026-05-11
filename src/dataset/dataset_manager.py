from pathlib import Path
from dataclasses import dataclass
from typing import List,Optional
import json

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
    auth_token: Optional[str] = None
    source_path:Optional[str] = None # 子路径下载  这个不太懂
    revision: Optional[str] = None # 版本号
    allow_patterns:Optional[List[str]] = None # 需要下载的文件过滤
    ignore_patterns:Optional[List[str]] = None # 不需要下载的文件过滤



    @property
    def full_local_path(self) -> Path:
        return DATASET_PATH / self.local_path

class DatasetDownloader:

    def __init__(self,dataset:Dataset):
        self.dataset = dataset
        self.path = self.dataset.full_local_path
        self.permitplatform = ['huggingface','ModelScope','']
        

    def execute(self):

        self.path.mkdir(parents=True,exist_ok=True)
        download_func = self._get_download_function()
        assert download_func is not None
        # TODO 对download_func为None时进行错误处理
        try:
            download_func(
                repo_id = self.dataset.name,
                local_dir = self.path,
                local_dir_use_symlinks=False,
            )
            pass
        except Exception as e:
            pass
        pass

    def _get_download_function(self):
        # TODO 加入模糊/近似匹配
        platform = self.dataset.platform
        if platform not in self.permitplatform:
            print('不支持该平台')
            print(f"当前支持的平台有{','.join(self.permitplatform)}")
            return None
        if platform == "huggingface":
            try:
                # from huggingface_hub import hf_hub_download
                from huggingface_hub import snapshot_download
                """参数参考
                snapshot_download(
                    repo_id="username/dataset_name",
                    repo_type="dataset",           # 仓库类型
                    revision="main",               # 版本分支/commit
                    local_dir="/path/to/save",     # 目标目录（⚠️ 注意语义）
                    cache_dir="/path/to/cache",    # 缓存目录
                    allow_patterns=["*.json"],     # 只下载匹配的文件
                    ignore_patterns=["*.pt"],      # 忽略的文件
                    token="hf_xxx",                # 认证token（私有数据集需要）
                    resume_download=True,          # 断点续传
                    local_dir_use_symlinks=False,  # 不使用软链接
                )"""
                return snapshot_download
            except ImportError as e:
                print(f"缺少huggingface_hub，使用pip install huggingface_hub 进行下载")
        elif platform == "ModelScope":
            try:
                from modelscope import snapshot_download
                return snapshot_download
            except ImportError as e:
                print(f"缺少modelscope库，请使用pip install modelscope进行安装")
        elif platform == "Kaggle":
            try:
                from kagglehub import dataset_download
                return dataset_download
            except ImportError as e:
                print(f"缺少kagglehub库，请使用pip install kagglehub进行安装")
        else:
            def fun(**kwargs):
                print('意外情况')
                return
            return fun


class DatasetLoader:
    def __init__(self,path:Path):
        self.path = DATASET_PATH
    # TODO 



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
    def add(self,dataset:Dataset,downloadnow:bool = True):
        if any(d.name ==dataset.name for d in self.datasets):
            raise ValueError(f"数据集{dataset.name}已存在")
        self.datasets.append(dataset)
        self._save()
        if downloadnow:
            downloader = DatasetDownloader(dataset = dataset)
            downloader.execute()
    
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
        # name作为键值，不得重复，不得修改
        ds = self.get(name)
        if ds is None:
            raise KeyError(f"数据集{name}不存在")
        for key,value in kwargs.items():
            if hasattr(ds,key):
                setattr(ds,key,value)
        self._save()
    
    def list(self,enabled_only:bool=False):
        if enabled_only:
            return [d for d in self.datasets if d.enabled]
        return self.datasets.copy()

    def verify(self):
        missing = []
        for d in self.datasets:
            print(f"{d.name}\t\t\t已下载" if d.full_local_path.exists() else f"{d.name}\t\t\t不存在")
            if not d.full_local_path.exists():
                missing.append(d)        
        return missing
    

