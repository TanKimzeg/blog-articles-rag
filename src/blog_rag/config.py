from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

ROOT_DIR = Path(__file__).resolve().parents[2]

@dataclass
class BlogRAGConfig:
    data_dir: Path = ROOT_DIR / "resources"
    index_dir: Path = ROOT_DIR / "resources" / "vector_index"
    # 模型配置
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = "deepseek-chat"

    # 检索配置
    top_k: int = 3

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BlogRAGConfig':
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_dir": str(self.data_dir),
            "index_dir": str(self.index_dir),
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
    
DEFAULT_CONFIG = BlogRAGConfig()