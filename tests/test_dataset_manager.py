# tests/test_dataset_manager.py
import json
from unittest.mock import patch,MagicMock

import pytest

from dataset.dataset_manager import Dataset,DatasetDownloader,DatasetManager


# 数据准备

@pytest.fixture
def sample_dataset():
    return Dataset(
        name="test_data",
        displayname="测试数据集",
        platform="huggingface",
        remote_path="Congliu/Chinese-DeepSeek-R1-Distill-data-110k",
        local_path="test_data",
        enabled=True,
        allow_patterns=["*.txt"]
    )

# 测试DatasetDownloader

def test_unsupported_platform(sample_dataset,capsys):
    sample_dataset.platform = "unknown"
    downloader = DatasetDownloader(sample_dataset)
    func = downloader._get_download_function()
    assert func is None
    captured = capsys.readouterr()
    assert "不支持该平台" in captured.out


@patch("importlib.import_module")
def test_huggingface_adapter(mock_import,sample_dataset):
    mock_module = MagicMock()
    mock_import.return_value = mock_module
    mock_module.snapshot_download = MagicMock()

    downloader = DatasetDownloader(sample_dataset)
    warpper = downloader._get_download_function()
    warpper()

    mock_module.snapshot_download.assert_called_once_with(
        repo_id = "Congliu/Chinese-DeepSeek-R1-Distill-data-110k",
        local_dir = sample_dataset.full_local_path,
        local_dir_use_symlinks = False,
        allow_patterns=["*.txt"],
        ignore_patterns = None,
        force_download = False,
    )

@patch("importlib.import_module")
def test_modelscope_adapter(mock_import,sample_dataset):
    sample_dataset.platform = "ModelScope"
    mock_module = MagicMock()
    mock_import.return_value = mock_module
    mock_module.snapshot_download = MagicMock()

    downloader = DatasetDownloader(sample_dataset)
    warpper = downloader._get_download_function()
    warpper()

    mock_module.snapshot_download.assert_called_once_with(
        model_id = "Congliu/Chinese-DeepSeek-R1-Distill-data-110k",
        cache_dir = str(sample_dataset.full_local_path)
    )

@patch("importlib.import_module")
def test_kaggle_adapter(mock_import,sample_dataset):
    sample_dataset.platform = "Kaggle"
    mock_module = MagicMock()
    mock_import.return_value = mock_module
    mock_module.dataset_download = MagicMock(return_value = "/fake/path")

    downloader = DatasetDownloader(sample_dataset)
    warpper = downloader._get_download_function()
    result = warpper()
    mock_module.dataset_download.assert_called_once_with(
        path = "Congliu/Chinese-DeepSeek-R1-Distill-data-110k",
        output_dir = sample_dataset.full_local_path,
        force = False
    )
    assert result == "/fake/path"

# DatasetManager 测试

def test_manager_add_and_save(sample_dataset,mock_dataset_path):
    mgr = DatasetManager()
    mgr._storage_file = mock_dataset_path / "datasets.json"
    mgr.datasets.clear()
    mgr._save()

    mgr.add(sample_dataset,downloadnow=False)

    # 内存检查
    assert len(mgr.datasets) == 1
    assert mgr.get("test_data") == sample_dataset

    # 文件检查
    storage_file = mock_dataset_path/"datasets.json"
    assert storage_file.exists()
    with open(storage_file,'r', encoding='utf-8') as f:
        data = json.load(f)

    assert data[0]['name'] == "test_data"

def test_manager_remove(sample_dataset):
    mgr = DatasetManager()
    mgr.datasets.clear()
    mgr._save()
    mgr.add(sample_dataset,downloadnow=False)
    mgr.remove("test_data")
    assert mgr.get("test_data") is None

def test_manager_update(sample_dataset):
    mgr = DatasetManager()
    mgr.datasets.clear()
    mgr._save()
    mgr.add(sample_dataset,downloadnow=False)
    mgr.update("test_data",displayname="新名字",enabled = False)
    ds = mgr.get("test_data")
    assert ds.displayname == "新名字"
    assert ds.enabled is False

def test_manager_list(sample_dataset):
    mgr = DatasetManager()
    mgr.datasets.clear()
    mgr._save()
    mgr.add(sample_dataset, downloadnow=False)
    d2 = Dataset(name="disabled_ds", displayname="", platform="hf",
                 remote_path="a", local_path="b", enabled=False)
    mgr.add(d2, downloadnow=False)

    all_ds = mgr.list(enabled_only=False)
    assert len(all_ds) == 2
    enabled_ds = mgr.list(enabled_only=True)
    assert len(enabled_ds) == 1
    assert enabled_ds[0].name == "test_data"

# 测试verify逻辑 模拟用户输入y
def test_verify_missing_dataset(sample_dataset,monkeypatch,mock_dataset_path):
    mgr = DatasetManager()
    mgr._storage_file = mock_dataset_path / "datasets.json"
    mgr.datasets.clear()
    mgr._save()
    mgr.add(sample_dataset,downloadnow=False)

    monkeypatch.setattr("builtins.input",lambda _: "y")
    with patch.object(DatasetDownloader,"execute") as mock_execute:
        missing = mgr.verify()
        mock_execute.assert_called_once()
        assert len(missing) == 1
        assert missing[0].name == "test_data"
