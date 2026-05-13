import os
from pathlib import Path
from dotenv import load_dotenv


# 获得项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 检查一遍路径
print(PROJECT_ROOT)
# exit()

# 指定.env文件
dotenv_path = PROJECT_ROOT/".env"
load_dotenv(dotenv_path=dotenv_path)

def get_abs_path(env_key:str)->Path:
    rel_path = os.getenv(env_key)
    if rel_path is None:
        raise ValueError(f"环境变量{env_key}未定义")
    return (PROJECT_ROOT/rel_path).resolve()

# 导出绝对路径
ROOT_DIR = get_abs_path("ROOT_DIR")
CACHE_PATH = get_abs_path("CACHE_PATH")
DATASET_PATH = get_abs_path("DATASET_PATH")
