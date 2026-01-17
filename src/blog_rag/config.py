import os
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Any, Dict
import json


ROOT_DIR = Path(__file__).resolve().parents[2]

class BlogRAGConfig(BaseSettings):
    """应用配置
    """
    renew: bool = Field(default=False, description="是否重新构建知识向量索引")
    index_dir: Path = Field(default=ROOT_DIR / "resources" / "vector_index", description="向量索引目录")
    markdown_dir: Path = Field(default=ROOT_DIR / "resources" / "markdown", description="Markdown 文件目录")
    cache_dir: Path = Field(default=ROOT_DIR / "resources" / "cache", description="切分数据的保存路径")

    # 模型配置
    embedding_model: str = Field(default="BAAI/bge-small-zh-v1.5", description="嵌入模型标识")
    llm_model: str = Field(default="deepseek-chat", description="生成模型标识")
    api_key: Optional[str] = Field(default=None, description="DeepSeek API 密钥")

    # 检索配置
    top_k: int = Field(default=10, ge=1, description="检索返回的默认top_k")

    # 生成配置
    temperature: float = Field(default=0.1, ge=0.0, le=1.0, description="温度参数")
    max_tokens: int = Field(default=2048, ge=1, description="最大生成长度")

    model_config = SettingsConfigDict(
        env_file=".env"
    )


# 兼容导出，默认从环境和 .env 加载（如需从文件加载，请显式调用 load_config(path)）
DEFAULT_CONFIG = BlogRAGConfig()