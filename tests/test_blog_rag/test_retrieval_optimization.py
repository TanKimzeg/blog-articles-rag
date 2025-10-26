from pathlib import Path
from blog_rag.rag_modules import DataPreparationModule, IndexConstructionModule, RetrievalOptimizationModule


def _prepare_vectorstore(sample_md_dir: Path):
    dpm = DataPreparationModule(data_dir=sample_md_dir)
    docs = dpm.load_markdowns()
    dpm.documents = docs
    chunks = dpm.chunk_markdowns()

    icm = IndexConstructionModule(index_save_path=sample_md_dir / "vector_index")
    vs = icm.build_vector_index(chunks)
    return vs, chunks


def test_hybrid_search(sample_md_dir: Path):
    vs, chunks = _prepare_vectorstore(sample_md_dir)
    rom = RetrievalOptimizationModule(vectorstore=vs, chunks=chunks)

    res = rom.hybrid_search("dropout", top_k=3)
    assert len(res) <= 3
    assert any("dropout" in d.page_content for d in res)


def test_metadata_filtered_search(sample_md_dir: Path):
    vs, chunks = _prepare_vectorstore(sample_md_dir)
    rom = RetrievalOptimizationModule(vectorstore=vs, chunks=chunks)

    # 过滤条件：用我们建议的标量键 category 或布尔 tag_llm
    res = rom.metadata_filtered_search("dropout", filters={"category": "tech"}, top_k=3)
    assert isinstance(res, list)
