import os
from pathlib import Path
import pytest

from blog_rag.rag_modules import DataPreparationModule


@pytest.fixture(autouse=True)
def _reset_class_state():
    # 重置类级状态，避免测试间相互影响
    DataPreparationModule.CATEGORIES.clear()
    DataPreparationModule.TAGS.clear()
    DataPreparationModule.id2markdown.clear()
    yield


@pytest.fixture
def md_sample(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    md_dir = data_dir / "markdown"
    md_dir.mkdir(parents=True, exist_ok=True)

    content = (
        "---\n"
        "title: 实现注意力机制 | LLM\n"
        "description: 从零构建大语言模型第3章:本章讲解注意力机制\n"
        "pubDate: 2025 09 22\n"
        "categories:\n"
        "  - tech\n"
        "tags:\n"
        "  - llm\n"
        "---\n\n"
        "# 实现注意力机制\n\n"
        "## 使用因果注意力机制来屏蔽后续词\n"
        "段落A\n\n"
        "### 使用`dropout`遮掩额外的 注意力权重\n"
        "段落B\n\n"
        "## 其它\n"
        "段落C\n"
    )
    md_path = md_dir / "sample.md"
    md_path.write_text(content, encoding="utf-8")
    return data_dir


def test_load_markdowns_parses_front_matter(md_sample: Path):
    dpm = DataPreparationModule(data_dir=md_sample)
    docs = dpm.load_markdowns()

    assert len(docs) == 1
    doc = docs[0]

    # front matter 已被解析进 metadata
    assert doc.metadata.get("title") == "实现注意力机制 | LLM"
    assert doc.metadata.get("categories") == ["tech"]
    assert doc.metadata.get("tags") == ["llm"]

    # id 映射已建立
    file_id = doc.metadata.get("file_id")
    assert file_id
    assert file_id in dpm.id2markdown

    # 汇总集合
    cats, tags = dpm.get_categories_and_tags()
    assert "tech" in cats
    assert "llm" in tags


def test_chunk_markdowns_splits_and_propagates_metadata(md_sample: Path):
    dpm = DataPreparationModule(data_dir=md_sample)
    docs = dpm.load_markdowns()

    # NOTE: 当前实现未在 load_markdowns() 中设置 self.documents，这里显式赋值
    dpm.documents = docs

    chunks = dpm.chunk_markdowns()
    assert len(chunks) > 0

    # 每个块都应带有父文档与块级元数据
    for i, c in enumerate(chunks):
        assert c.metadata.get("parent_id") == docs[0].metadata.get("file_id")
        assert isinstance(c.metadata.get("chunk_id"), str)
        assert isinstance(c.metadata.get("chunk_index"), int)
        assert isinstance(c.metadata.get("chunk_size"), int)
        # 继承 front matter
        assert c.metadata.get("title") == "实现注意力机制 | LLM"

    # 至少应切出多个块（含不同层级标题）
    assert any("使用因果注意力机制" in c.page_content for c in chunks)
    assert any("dropout" in c.page_content for c in chunks)

