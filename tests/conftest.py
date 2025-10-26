from pathlib import Path
import pytest


@pytest.fixture
def sample_md_dir() -> Path:
    """返回测试样例数据根目录（其下包含 markdown/ 子目录）。"""
    return Path(__file__).resolve().parent / "fixtures" / "data"

