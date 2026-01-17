import uuid
import logging
import asyncio
from fastapi import FastAPI, APIRouter, Depends, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request as StarletteRequest
from starlette import status

from blog_rag import BlogRAGSystem
from api.schemas import ApiResponse, ok, fail

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Blog RAG System (deferred build)...")

    # 创建 RAG 系统但不在构造时自动启动耗时操作
    rag = BlogRAGSystem(auto_start=False)
    rag.initialize_modules()
    app.state.rag = rag
    try:
        yield
    finally:
        # 在关闭时尝试优雅取消后台构建任务
        task = getattr(app.state, "rag_build_task", None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.info("后台索引构建任务已取消")
        # 在此释放其他资源（如需要）
        logger.info("Blog RAG System shutdown complete.")


app = FastAPI(title="Blog RAG API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 统一依赖：从 state 读取 RAG，若未初始化则惰性创建（测试/生产均可用）
async def get_rag_dep(request: Request) -> BlogRAGSystem:
    rag = getattr(request.app.state, "rag", None)
    if rag is None:
        logger.warning("app.state.rag 未初始化，进行惰性创建（sync/测试场景）。")
        rag = BlogRAGSystem(auto_start=False)
        await asyncio.to_thread(rag.initialize_modules)
        request.app.state.rag = rag
        return rag

    # 如果有正在进行的构建任务，等待其完成以保证检索可用
    build_task = getattr(request.app.state, "rag_build_task", None)
    if build_task and not build_task.done():
        logger.info("等待后台索引构建完成...")
        try:
            await build_task
        except Exception as e:
            logger.warning("后台索引构建任务失败: %s", e)

    return rag


# 我已经被Java毒害了😭
class ChunkVO(BaseModel):
    content: str
    metadata: Dict[str, Any]

class MarkdownVO(BaseModel):
    content: str
    metadata: Dict[str, Any]
    path: str

class FilterDTO(BaseModel):
    categories: Optional[List[str]] = None
    tags: Optional[List[str]] = None

class SearchDTO(BaseModel):
    query: str
    topK: int = 10
    page: int = 1
    size: int = 10
    filters: Optional[FilterDTO] = None
    highlight: bool = False

class PageResult(BaseModel):
    items: List[ChunkVO]
    total: int
    page: int
    size: int


# 版本化路由
api_v1 = APIRouter()


@api_v1.get("/meta/categories", response_model=ApiResponse[Dict[str, Any]])
def v1_categories(rag: BlogRAGSystem = Depends(get_rag_dep)):
    categories = list(rag.data_module.categories) if rag.data_module else []
    return ok(data={"items": categories, "total": len(categories)})

@api_v1.get("/meta/tags", response_model=ApiResponse[Dict[str, Any]])
def v1_tags(rag: BlogRAGSystem = Depends(get_rag_dep)):
    tags = list(rag.data_module.tags) if rag.data_module else []
    return ok(data={"items": tags, "total": len(tags)})

@api_v1.post("/search", response_model=ApiResponse[PageResult])
def v1_search(payload: SearchDTO = Body(...), rag: BlogRAGSystem = Depends(get_rag_dep)):
    q = payload.query
    k = payload.topK or payload.size or 10
    filters: Optional[Dict[str, Any]] = None
    if payload.filters and (payload.filters.categories or payload.filters.tags):
        filters = {}
        if payload.filters.categories:
            filters["categories"] = {"$gte": payload.filters.categories}
        if payload.filters.tags:
            filters["tags"] = {"$gte": payload.filters.tags}
    chunks = rag.query_chunks(q, filters, k)
    items = [ChunkVO(content=c.content, metadata=c.metadata) for c in chunks]
    return ok(data=PageResult(items=items, total=len(items), page=payload.page, size=payload.size))

@api_v1.get("/docs/{doc_id}", response_model=ApiResponse[MarkdownVO])
def v1_get_doc(doc_id: str, rag: BlogRAGSystem = Depends(get_rag_dep)):
    md = rag.query_markdown(doc_id)
    if not md:
        return fail(message="Document not found")
    return ok(data=MarkdownVO(content=md.content, metadata=md.metadata, path=str(md.path)))

# 注册 v1 路由
app.include_router(api_v1)

@app.middleware("http")
async def add_trace_id(request: StarletteRequest, call_next):
    request.state.trace_id = uuid.uuid4().hex
    resp = await call_next(request)
    resp.headers["X-Request-ID"] = request.state.trace_id
    return resp

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = ApiResponse[dict](success=False, code=40001, message="validation error",
                             data={"errors": exc.errors()})
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=body.model_dump())

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    body = ApiResponse[dict](success=False, code=50000, message="internal error",
                             data=None)
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=body.model_dump())