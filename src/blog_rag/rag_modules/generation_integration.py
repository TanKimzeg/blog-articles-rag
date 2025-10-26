import os
import logging
from typing import Any, List
from pydantic import SecretStr

from langchain_deepseek import ChatDeepSeek
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

CHUNKS = List[Document]

logger = logging.getLogger(__name__)

class BasicChatModel:
    llm: ChatDeepSeek
    def __init__(
            self,
            model_name: str,
            api_key: str,
            temperature: float = 0.0,
            max_tokens: int = 2048,
        ) -> None:
        if 'deepseek' in model_name.lower():
            self.llm = ChatDeepSeek(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=SecretStr(api_key),
            )
        else:
            raise ValueError(f"不支持的模型名称: {model_name}")
        self.basic_prompt_template = ChatPromptTemplate.from_template(
"""
基于以下信息，回答用户的问题。

相关信息:
{context}

用户问题: {question}

请提供详细、实用的回答。如果信息不足，请诚实说明。

回答:
"""
        )
        logger.info("聊天模型初始化完成!")

    def generate_answer(self, prompt: str) -> str:
        """生成回答"""
        response = self.llm.invoke(prompt)
        return response.text
        
class GenerationIntegrationModule(BasicChatModel):
    """生成集成模块 - 负责LLM集成和回答生成"""
    def __init__(
            self,
            model_name: str = "deepseek-chat",
            temperature: float = 0.0,
            max_tokens: int = 2048,
        ) -> None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY 环境变量未设置。请设置您的 DeepSeek API 密钥。")
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate_basic_answer(self, question: str, context: CHUNKS) -> str:
        """基于上下文生成回答"""
        context_text = self._build_context(context)
        chat_prompt = self.basic_prompt_template.format_prompt(context=context_text, question=question)
        return self.generate_answer(chat_prompt.to_string())
    
    def _build_context(self, chunks: CHUNKS, max_length: int = 1000) -> str:
        """构建上下文字符串"""
        if not chunks:
            return "暂无相关信息。"
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(chunks, 1):
            # 添加元数据信息
            metadata_info = f"[分块 {i}]"
            if 'h1' in doc.metadata:
                metadata_info += f"一级标题: {doc.metadata['h1']}"
            if 'h2' in doc.metadata:
                metadata_info += f"| 二级标题: {doc.metadata['h2']}"
            if 'h3' in doc.metadata:
                metadata_info += f"| 三级标题: {doc.metadata['h3']}"
            if 'h4' in doc.metadata:
                metadata_info += f" | 四级标题: {doc.metadata['h4']}"
            if 'categories' in doc.metadata:
                metadata_info += f" | 分类: {doc.metadata['categories']}"
            if 'tags' in doc.metadata:
                metadata_info += f" | 标签: {doc.metadata['tags']}"

            # 构建文档文本
            doc_text = f"{metadata_info}\n{doc.page_content}\n"
            
            # 检查长度限制
            if current_length + len(doc_text) > max_length:
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "\n" + "="*20 + "\n".join(context_parts)