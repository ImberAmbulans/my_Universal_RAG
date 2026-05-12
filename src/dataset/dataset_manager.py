from pathlib import Path
from dataclasses import dataclass
from typing import List,Optional
import json
import importlib
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
    _PLATFORM_REGISTRY={
        "huggingface": ("huggingface_hub", "snapshot_download",None),
        "ModelScope": ("modelscope", "snapshot_download",None),
        "Kaggle": ("kagglehub", "dataset_download",None),
    }


    def __init__(self,dataset:Dataset):
        self.dataset = dataset
        self.path = self.dataset.full_local_path
        self.permitplatform = ['huggingface','ModelScope','Kaggle']
        

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
        module_name,function_name,adapter = self._PLATFORM_REGISTRY.get(platform)
        try:
            module = importlib.import_module(module_name)
            raw_func = getattr(module,function_name)
        except ImportError as e:
            # TODO log
            print(f'{module_name}导入失败 尝试使用pip install {module_name}')
        if adapter is None:
            adapter = self._make_default_adapter(raw_func)
        def warpper(**kwargs):
            return adapter(**kwargs)
        return warpper




    def _make_default_adapter(self,raw_func):
        def adapter(repo_id:str,local_dir:str,**extra):
            return raw_func(
                repo_id = repo_id,
                local_dir = local_dir,
                **extra
                )
        return adapter
    
    # 为不同平台编写特定适配器
    @staticmethod
    def _adapt_modelscope(raw_func):
        def adapter(repo_id, local_dir, **extra):
        # modelscope 使用 cache_dir，且没有 local_dir_use_symlinks
            return raw_func(model_id=repo_id, cache_dir=str(local_dir))
        return adapter

    @staticmethod
    def _adapt_kaggle(raw_func):
        def adapter(repo_id, local_dir, **extra):
                # kagglehub.dataset_download 参数为路径字符串，返回下载路径
                # 需要额外处理（具体 API 可能变化，此处示例）
            downloaded_path = raw_func(repo_id)
                # 可能需要软链接或移动文件到 local_dir，这里简化
            return downloaded_path
        return adapter

# 更新注册表，为 ModelScope 和 Kaggle 指定适配器
DatasetDownloader._PLATFORM_REGISTRY["ModelScope"] = ("modelscope", "snapshot_download", DatasetDownloader._adapt_modelscope)
DatasetDownloader._PLATFORM_REGISTRY["Kaggle"] = ("kagglehub", "dataset_download", DatasetDownloader._adapt_kaggle)    

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
    

