import logging
from typing import Any, Dict, List
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
    def __init__(
        self,
        config: BlogRAGConfig | None = None,
        data_module=None,
        index_module=None,
        retrieval_module=None,
        generation_module=None,
        auto_start: bool = True,
    ):
        """可注入模块的 RAG 系统。

        参数说明：
        - config: 配置
        - data_module / index_module / retrieval_module / generation_module: 可注入的模块实例（可为 None，系统会按需创建）
        - auto_start: 是否在构造时自动调用 initialize_modules 与 build_knowledge_index（默认 True，以兼容原行为）
        """
        self.config = config or DEFAULT_CONFIG
        self.data_module = data_module
        self.index_module = index_module
        self.retrieval_module = retrieval_module
        self.generation_module = generation_module

        logger.info("BlogRAGSystem 创建，auto_start=%s", auto_start)

        if auto_start:
            self.initialize_modules()
            self.build_knowledge_index()

    def initialize_modules(self):
        '''初始化各个模块（若调用者已注入则复用）'''
        logger.info("正在初始化RAG系统...")

        # 1. 数据准备模块
        if self.data_module is None:
            logger.info("正在初始化数据准备模块...")
            self.data_module = DataPreparationModule(
                markdown_dir=self.config.markdown_dir,
                cache_dir=self.config.cache_dir
            )
        else:
            logger.info("使用注入的数据准备模块。")

        # 2. 索引构建模块
        if self.index_module is None:
            logger.info("正在初始化索引构建模块...")
            self.index_module = IndexConstructionModule(
                model_name=self.config.embedding_model,
                index_save_path=self.config.index_dir
            )
        else:
            logger.info("使用注入的索引构建模块。")

        # 3. 生成集成模块
        if self.generation_module is None:
            logger.info("正在初始化生成集成模块...")
            self.generation_module = GenerationIntegrationModule(
                model_name=self.config.llm_model,
                api_key=self.config.api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        else:
            logger.info("使用注入的生成模块。")

        # 4. 检索优化模块
        if self.retrieval_module is None:
            logger.info("正在初始化检索优化模块...")
            self.init_retrieval_module()
        else:
            logger.info("使用注入的检索优化模块。")

        logger.info("模块初始化完成")

    def load_knowledge_index(self) -> bool:
        '''加载知识向量索引，返回是否成功'''
        try:
            assert self.index_module is not None
            assert self.data_module is not None
            logger.info("正在加载和处理文档...")
            chunks = self.data_module.load_chunks()
            vectorstore = self.index_module.load_vector_index()
            if vectorstore is not None:
                logger.info("成功加载已保存的向量索引!")
            else:
                raise RuntimeError("未找到已保存的向量索引，无法加载知识索引。")
            self.retrieval_module = RetrievalOptimizationModule(
                vectorstore=vectorstore,
                chunks=chunks
            )
            return True
        except Exception as e:
            logger.error(f"构建知识向量索引失败: {e}")
            return False

    def build_knowledge_index(self) -> bool:
        '''加载知识向量索引，返回是否成功'''
        try:
            assert self.index_module is not None
            assert self.data_module is not None
            logger.info("正在加载和处理文档...")
            _, chunks = self.data_module.renew_data()
            vectorstore = self.index_module.build_vector_index(chunks)
            logger.info("正在保存向量索引...")
            self.index_module.save_vector_index(vectorstore)
            self.retrieval_module = RetrievalOptimizationModule(
                vectorstore=vectorstore,
                chunks=chunks
            )
            return True
        except Exception as e:
            logger.error(f"构建知识向量索引失败: {e}")
            return False

    def init_retrieval_module(self) -> bool:
        logger.info("初始化检索优化")
        # 检查路径，早期错误更明确，避免在构造时 exit()
        if not Path(self.config.index_dir).exists():
            logger.warning(f"索引目录不存在: {self.config.index_dir}")
            Path(self.config.index_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"已创建索引目录: {self.config.index_dir}")
        if not Path(self.config.cache_dir).exists():
            logger.warning(f"缓存目录不存在: {self.config.cache_dir}")
            Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"已创建缓存目录: {self.config.cache_dir}")
        if not Path(self.config.markdown_dir).exists():
            logger.warning(f"Markdown目录不存在: {self.config.markdown_dir}")

        assert self.data_module is not None
        if self.retrieval_module is not None:
            logger.info("检索模块已初始化，跳过重复初始化。")
            return True
        if self.config.renew:
            return self.build_knowledge_index()
        else:
            return self.load_knowledge_index()

    def query_chunks(
            self, 
            query: str, 
            filters: Dict[str, Any] | None,
            top_k: int
        ) -> List[ChunkInfo]:
        assert self.retrieval_module is not None
        logger.info("正在执行查询...")
        if filters is None:
            relevant_chunks = self.retrieval_module.hybrid_search(
                query, top_k
            )
        else:
            relevant_chunks = self.retrieval_module.metadata_filtered_search(
                query, filters, top_k
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
        logger.warning(f"未找到ID为 {id} 的Markdown文档。")
        return None
