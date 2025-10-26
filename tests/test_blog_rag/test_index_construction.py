from pathlib import Path
from blog_rag.rag_modules import DataPreparationModule, IndexConstructionModule


def test_build_and_save_load_faiss(sample_md_dir: Path):
    dpm = DataPreparationModule(data_dir=sample_md_dir)
    docs = dpm.load_markdowns()
    dpm.documents = docs
    chunks = dpm.chunk_markdowns()

    icm = IndexConstructionModule(index_save_path=sample_md_dir / "vector_index")
    vectorstore = icm.build_vector_index(chunks)
    assert vectorstore is not None

    icm.save_vector_index(vectorstore)

    # 重新加载
    icm2 = IndexConstructionModule(index_save_path=sample_md_dir / "vector_index")
    icm2.setup_embeddings()
    loaded = icm2.load_vector_index()
    assert loaded is not None


def test_similarity_search_with_filter(sample_md_dir: Path):
    dpm = DataPreparationModule(data_dir=sample_md_dir)
    docs = dpm.load_markdowns()
    dpm.documents = docs
    chunks = dpm.chunk_markdowns()

    icm = IndexConstructionModule(index_save_path=sample_md_dir / "vector_index")
    icm.build_vector_index(chunks)

    # 简单无过滤
    res = icm.similarity_search("dropout", k=2)
    assert len(res) > 0

    # 过滤（简单等值）不应抛错
    _ = icm.similarity_search("dropout", k=2, filter={"categories": {"$gte": ["tech"]}})
