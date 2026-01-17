# 博客文章智慧搜索（RAG）

一个基于 LangChain + FAISS 的轻量级博客知识检索与生成（RAG）项目：

- 后端：FastAPI（统一响应、可重建索引）
- 前端：Vite + Vue3 + Vue Router（支持独立文档页 /view/:id，Markdown 图片自适应）

## 项目结构

```text
my-rag/
├─ resources/
│  ├─ markdown/                 # 你的 Markdown 源文档
│  └─ vector_index/             # 向量索引（FAISS 持久化）
├─ src/
│  ├─ api/
│  │  └─ app.py                 # FastAPI 入口：/search /docs/{id} /meta/tags /meta/categories /reindex
│  └─ blog_rag/
│     ├─ main.py                # RAG 系统装配（数据/索引/检索/生成）
│     ├─ config.py              # 路径与参数配置（data_dir/index_dir/模型等）
│     └─ rag_modules/
│        ├─ data_preparation.py
│        ├─ index_construction.py
│        ├─ retrieval_optimization.py
│        └─ generation_integration.py
├─ frontend/
│  ├─ index.html
│  ├─ vite.config.ts            # 开发环境代理 /api -> http://localhost:8000
│  ├─ package.json
│  └─ src/
│     ├─ App.vue                # 外壳（<router-view/>）
│     ├─ main.ts                # 应用入口
│     ├─ router.ts              # 路由：/（搜索）/view/:id（文档）
│     ├─ pages/Home.vue         # 搜索页
│     └─ view/index.vue         # 文档页（Markdown 渲染、图片自适应）
├─ tests/
│  └─ ...
├─ pyproject.toml
└─ README.md
```

Python 版本：pyproject 要求 `>= 3.13`

## 环境与依赖

1. 安装后端依赖（使用 uv）：

   ```shell
   uv sync
   uv pip install -e .
   ```

2. 安装前端依赖（Node 18+ 推荐）：

   ```shell
   cd frontend
   npm install
   ```

## 数据准备

在 `.env` 中可进行自定义配置，包括Markdown文件所在目录、是否重建索引、API Key等。

Markdown文件建议带 YAML front matter，如：

```yaml
---
title: 注意力机制
categories: [tech]
tags: [llm]
---
# 正文...
```

## 启动

后端（FastAPI）：

- `uv run uvicorn api.app:app --reload`

前端（Vite 开发服务器，已代理 /api 到 8000）：

- cd frontend; npm run dev
- 打开 <http://127.0.0.1:5173>

生产构建与本地预览：

- cd frontend; npm run build; npm run preview
- 打开 <http://127.0.0.1>

环境变量（前端）：

- 默认 API 基址为 /api（见 `src/view/Index.vue` 与 `src/pages/Home.vue`）。
- 开发环境已通过 `vite.config.ts` 把 /api 代理到 <http://localhost:8000>。
- 若不使用代理，可在 .env 中设置 VITE_API_BASE，例如：`VITE_API_BASE=http://127.0.0.1:8000`

## API 说明（统一响应）

所有业务接口返回统一响应体：

```json
{
  "success": true,
  "code": 0,
  "message": "ok",
  "data": {},
  "traceId": "请求ID，可选",
  "ts": 1710000000000
}
```

- GET `/meta/categories`

  - 返回：`data.items` 为分类数组，`data.total` 为数量。

- GET `/meta/tags`

  - 返回：`data.items` 为标签数组，`data.total` 为数量。

- POST `/search`

  - 请求体：

    ```json
    {
      "query": "注意力机制",
      "topK": 10,
      "page": 1,
      "size": 10,
      "filters": { "categories": ["tech"], "tags": ["llm"] },
      "highlight": false
    }
    ```

  - 返回：`data.items` 为文档块数组（每项包含 `content` 与 `metadata`）。

- GET `/docs/{doc_id}`

  - 返回整篇 Markdown：`content`、`metadata`、`path`。

说明：FAISS 原生过滤有限，复杂过滤以“先检索后过滤”为主，或替换为支持表达式过滤的向量库（如 Chroma/Milvus）。

## 前端特性

- 搜索页（/）：提供关键词 + 分类/标签过滤；“查看全文”跳转到 /view/:id。
- 文档页（/view/:id）：Markdown 渲染。

## 测试

- 安装测试依赖：`uv sync --group dev`
- 运行：`uv run pytest -q`
