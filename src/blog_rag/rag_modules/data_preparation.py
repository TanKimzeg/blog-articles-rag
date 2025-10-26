import re
import logging
import operator
from hashlib import md5
from pathlib import Path
from uuid import uuid4
from typing import Tuple, Dict, List, Set, Any
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

CHUNKS = List[Document]
MARKDOWNS = List[Document]

class DataPreparationModule:
    CATEGORIES: Set[Any] = set()
    TAGS: Set[Any] = set()
    id2markdown: Dict[str, Tuple[Path, Document]] = dict()

    def __init__(self, data_dir: str | Path):
        self.data_root = Path(data_dir).resolve()
        self.documents: MARKDOWNS = [] # 存储加载的 Markdown 文档列表
        self.chunks: CHUNKS = []      # 存储切分后的文档块列表

    def load_markdowns(self) -> MARKDOWNS:
        markdown_path = self.data_root / "markdown"
        logger.info(f"正在从 {markdown_path} 加载 Markdown 文件...")
        documents: MARKDOWNS = []
        for md_file in markdown_path.glob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            relative_path = md_file.relative_to(self.data_root).as_posix()
            file_id = md5(relative_path.encode("utf-8")).hexdigest()

            # 创建 Document 对象并附加元数据
            doc = Document(
                page_content=content,
                metadata={
                    "path": relative_path,
                    "file_id": file_id,
                    "doc_type": "markdown"
                }
            )
            documents.append(doc)
            self.id2markdown[file_id] = (md_file, doc)
        for doc in documents:
            self._update_metadata(doc)
            if "categories" in doc.metadata:
                if isinstance(doc.metadata["categories"], list):
                    self.CATEGORIES.update(doc.metadata["categories"])
                else:
                    logger.warning(f"文档 {doc.metadata.get('path', '未知')} 的 categories 不是列表类型。")
            if "tags" in doc.metadata:
                if isinstance(doc.metadata["tags"], list):
                    self.TAGS.update(doc.metadata["tags"])
                else:
                    logger.warning(f"文档 {doc.metadata.get('path', '未知')} 的 tags 不是列表类型。")
        self.documents = documents
        return documents

    def _update_metadata(self, doc: Document) -> None:
        content = doc.page_content
        meta = doc.metadata

        if not content.startswith("---"):
            logger.warning(f"文档 {meta.get('path', '未知')} 缺少 front matter，跳过元数据解析。")
            return

        # 匹配从文首开始的 front matter 区块
        m = re.match(r"^---\s*\n(.*?)\n---\s*\n?", content, re.DOTALL)
        if m is None:
            logger.warning(f"文档 {meta.get('path', '未知')} front matter 格式不正确，跳过元数据解析。")
            return

        fm_text = m.group(1)

        try:
            import yaml  # 可选依赖
            inline_meta = yaml.safe_load(fm_text) or {}
            if not isinstance(meta, dict):
                inline_meta = {}
        except Exception:
            inline_meta = {}
        doc.metadata.update(inline_meta)

    @classmethod
    def get_categories_and_tags(cls) -> Tuple[Set[Any], Set[Any]]:
        return cls.CATEGORIES, cls.TAGS

    def chunk_markdowns(self) -> CHUNKS:
        '''
        使用 MarkdownHeaderTextSplitter 按标题层级切分文档，
        Returns:
            List[Document]: 切分后的文档块列表
        '''
        logging.info("正在切分 Markdown 文档...")
        chunks: CHUNKS = self._markdown_split()

        self.chunks = chunks
        logger.info(f"切分完成!共切分出 {len(chunks)} 个文档块。")
        return chunks

    def _markdown_split(self) -> CHUNKS:
        # 切分层级
        headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False, # 保留标题以便上下文理解
        )
        chunks: CHUNKS = []
        for doc in self.documents:
            md_chunks = markdown_splitter.split_text(doc.page_content)
            
            logger.info(f"文档 {doc.metadata.get('path', '未知')} 被切分为 {len(md_chunks)} 个块。")

            # 为每个块建立与父文档的联系
            for i, chunk in enumerate(md_chunks):
                child_id = uuid4().hex
                chunk.metadata.update(doc.metadata)  # 继承父文档元数据

                chunk.metadata.update({
                    "chunk_id": child_id,
                    "parent_id": doc.metadata.get("file_id"),
                    "chunk_index": i,
                    "chunk_size": len(chunk.page_content)
                })
            chunks.extend(md_chunks)
        
        return chunks

    def filter_doc_by_categories(self, categories: List[str]) -> MARKDOWNS:
        '''根据类别过滤已加载的 Markdown 文档'''
        filtered_docs = [
            doc for doc in self.documents
            if "categories" in doc.metadata and
               operator.ge(doc.metadata["categories"], categories)
        ]
        logger.info(f"根据类别过滤后，剩余 {len(filtered_docs)} 个文档。")
        return filtered_docs

    def filter_doc_by_tags(self, tags: List[str]) -> MARKDOWNS:
        '''根据类别过滤已加载的 Markdown 文档'''
        filtered_docs = [
            doc for doc in self.documents
            if "categories" in doc.metadata and
               any(tag in doc.metadata["categories"] for tag in tags)
        ]
        logger.info(f"根据标签过滤后，剩余 {len(filtered_docs)} 个文档。")
        return filtered_docs

    def get_statistics(self) -> Dict[str, Any]:
        '''获取数据准备模块的统计信息'''
        stats = {
            "total_markdown_files": len(self.documents),
            "total_chunks": len(self.chunks),
            "unique_categories": list(self.CATEGORIES),
            "unique_tags": list(self.TAGS),
            "avg_chunk_size": sum(len(chunk.page_content) 
                                  for chunk in self.chunks) / len(self.chunks) if self.chunks else 0
        }
        return stats
