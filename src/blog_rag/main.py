import sys
import logging
from typing import Any, Dict, List, Tuple
from pathlib import Path
from dataclasses import dataclass
from blog_rag.config import BlogRAGConfig, DEFAULT_CONFIG
from blog_rag.rag_modules import *


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(name)s - %(levelname)s] %(message)s",
    datefmt="%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

@dataclass
class BasicInfo:
    content: str
    metadata: Dict[str, Any]
@dataclass
class ChunkInfo(BasicInfo):
    pass

@dataclass
class MarkdownInfo(BasicInfo):
    path: Path
    pass


class BlogRAGSystem:
    def __init__(self, config: BlogRAGConfig | None = None):
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None

        # 检查路径
        if not Path(self.config.data_dir).exists():
            logger.warning(f"数据目录不存在: {self.config.data_dir}")
            sys.exit(-1)
        if not Path(self.config.index_dir).exists():
            logger.warning(f"索引目录不存在: {self.config.index_dir}")
            sys.exit(-1)
        
        self.initialize_modules()
        self.build_knowledge_index()

    def initialize_modules(self):
        '''初始化各个模块'''
        logger.info("正在初始化RAG系统...")

        # 1. 数据准备模块
        logger.info("正在初始化数据准备模块...")
        self.data_module = DataPreparationModule(data_dir=self.config.data_dir)

        # 2. 索引构建模块
        logger.info("正在初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_dir
        )

        # 3. 生成集成模块
        logger.info("正在初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        logger.info("RAG系统初始化完成!")

    def build_knowledge_index(self):
        '''构建知识向量索引'''
        assert self.index_module is not None
        logger.info("正在构建知识向量索引...")

        assert self.data_module is not None
        logger.info("正在加载和处理文档...")
        self.data_module.load_markdowns()
        logger.info("正在切分文档...")
        chunks = self.data_module.chunk_markdowns()
        vectorstore = self.index_module.load_vector_index()
        if vectorstore is not None:
            logger.info("成功加载已保存的向量索引!")
        else:
            logger.info("未找到已保存的向量索引，开始构建新索引...")
            vectorstore = self.index_module.build_vector_index(chunks)
            logger.info("正在保存向量索引...")
            self.index_module.save_vector_index(vectorstore)

        logger.info("初始化检索优化")
        self.retrieval_module = RetrievalOptimizationModule(
            vectorstore=vectorstore,
            chunks=chunks
        )

        # 显示统计信息
        stats = self.data_module.get_statistics()

    def query_chunks(self, query: str, filters: Dict[str, Any] | None) -> List[ChunkInfo]:
        assert self.retrieval_module is not None
        logger.info("正在执行查询...")
        if filters is None:
            relevant_chunks = self.retrieval_module.hybrid_search(
                query, top_k=self.config.top_k
            )
        else:
            relevant_chunks = self.retrieval_module.metadata_filtered_search(
                query, filters, top_k=self.config.top_k
            )
        logger.info(f"检索到 {len(relevant_chunks)} 个相关文档块。")
        
        return [
            ChunkInfo(content=doc.page_content, metadata=doc.metadata)
            for doc in relevant_chunks
        ]

    def query_markdown(self, id: str) -> MarkdownInfo | None:
        assert self.data_module is not None
        logger.info(f"正在查询Markdown文档，ID: {id}...")
        path, doc = self.data_module.id2markdown.get(id, (None, None))
        if path and doc:
            return MarkdownInfo(
                content=doc.page_content,
                metadata=doc.metadata,
                path=path
            )
        return None



