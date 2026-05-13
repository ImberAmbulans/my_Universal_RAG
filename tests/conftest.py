import sys
from pathlib import Path

# 找到src目录
sys.path.insert(0,str(Path(__file__).parent.parent/"src"))

import pytest
from config import DATASET_PATH  
# DATASET_PATH是真是路径，会在测试中替换，要自己写替换

@pytest.fixture(autouse=True)
def mock_dataset_path(monkeypatch,tmp_path):
    """自动将所有测试中的DATASET_PATH替换为临时目录"""
    fake_path = tmp_path/"datasets"
    fake_path.mkdir()
    monkeypatch.setattr("config.DATASET_PATH",fake_path)
    return fake_path