from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from api.app import app


@pytest.fixture
def sample_md_dir() -> Path:
    """返回测试样例数据根目录（其下包含 markdown/ 子目录）。"""
    return Path(__file__).resolve().parent / "fixtures" / "data"


@pytest.fixture
def client():
    """提供启用 lifespan 的 TestClient。"""
    with TestClient(app) as c:
        yield c

