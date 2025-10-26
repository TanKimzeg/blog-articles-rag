import logging
from typing import List
from pathlib import Path

from huggingface_hub import snapshot_download
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

CHUNKS = List[Document]
MARKDOWNS = List[Document]

logger = logging.getLogger(__name__)

class IndexConstructionModule:
    """索引构建模块 - 负责向量化和索引构建"""
    embeddings: HuggingFaceEmbeddings | None = None
    vectorstore: FAISS | None = None
    def __init__(
            self, 
            model_name: str = "BAAI/bge-small-zh-v1.5",
            index_save_path: str | Path = "./resources/vector_index"
        ) -> None:
        self.model_name = model_name
        self.index_save_path = Path(index_save_path)

    def setup_embeddings(self):
        logger.info(f"正在初始化嵌入模型: {self.model_name} ...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=snapshot_download(self.model_name),
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        logger.info("嵌入模型初始化完成。")

    def build_vector_index(self, chunks: CHUNKS) -> FAISS:
        logger.info("正在构建向量索引...")
        if not self.embeddings:
            self.setup_embeddings()

        assert self.embeddings is not None
        vectorstore = FAISS.from_documents(
            documents=chunks, embedding=self.embeddings)
        self.vectorstore = vectorstore
        return vectorstore

    def add_chunks(self, new_chunks: CHUNKS):
        if not self.vectorstore:
            self.build_vector_index(new_chunks)
            return
        logger.info(f"正在向向量索引中添加 {len(new_chunks)} 个新文档块...")
        self.vectorstore.add_documents(new_chunks)
        logger.info("新文档块添加完成。")

        
    def save_vector_index(self, vectorstore: FAISS) -> None:
        save_path: str
        if isinstance(vectorstore, FAISS):
            save_path = str(Path(self.index_save_path / "faiss_index").resolve())
            vectorstore.save_local(str(save_path))
        else:
            raise ValueError(f"不受支持的向量存储类型: {vectorstore.__class__.__name__}")
        logger.info(f"向量索引已保存到: {save_path}")

    def load_vector_index(self, db_type: str = "FAISS") -> FAISS | None:
        if not self.embeddings:
            self.setup_embeddings()
        assert self.embeddings is not None

        try:
            if db_type.upper() == "FAISS":
                load_path = str(Path(self.index_save_path / "faiss_index").resolve())
                self.vectorstore = FAISS.load_local(
                    load_path, 
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"已从 {load_path} 加载 FAISS 向量索引。")
            else:
                raise ValueError(f"不受支持的向量存储类型: {db_type}")
        finally:
            return self.vectorstore

    def similarity_search(
            self, 
            query: str, 
            k: int = 4, 
            filter: dict | None = None
        ) -> CHUNKS:
        if not self.vectorstore:
            raise ValueError("向量索引尚未加载或构建。请先调用 load_vector_index 或 build_vector_index 方法。")
        logger.info(f"正在执行相似度搜索，查询: '{query}'，返回前 {k} 个结果。")
        results: CHUNKS = []
        if isinstance(self.vectorstore, FAISS):
            results = self.vectorstore.similarity_search(
                query,
                k,
                filter
            )
        logger.info(f"相似度搜索完成，找到 {len(results)} 个结果。")
        return results