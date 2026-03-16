# Chat Agent Backend

最小可用的 Agent 问答系统后端项目。

当前仓库已经完成阶段 1 初始化工程，并持续按阶段补齐配置管理、持久化、基础 API、Agent
主链路、知识库接入、工具系统和 MCP 骨架。

## 当前技术栈

- Python 3.11
- FastAPI
- LangChain
- LangGraph
- PostgreSQL
- Redis
- pytest
- Ruff
- pre-commit
- Docker / docker-compose

## 已完成内容

- 初始化项目目录结构
- 初始化 `pyproject.toml`
- 初始化 `.env.example`
- 初始化 `docker-compose.yml`
- 初始化 `.pre-commit-config.yaml`
- 建立 FastAPI 入口
- 提供 `/health` 接口
- 建立核心配置、日志、异常体系
- 建立会话、消息、短期记忆、RAGFlow 映射的持久化层
- 建立基础单轮会话、消息查询和对话 API
- 接入真实 LLM 单轮对话链路
- 增加 OpenAI Chat Completions 兼容适配层，便于后续接入 Qwen 等兼容模型

## 目录结构说明

```text
app/
├── api/
│   └── v1/
├── clients/
├── core/
├── schemas/
├── services/
├── agent/
│   └── nodes/
├── memory/
├── knowledge/
│   └── ragflow/
├── tools/
│   └── builtin/
├── mcp/
├── persistence/
└── main.py
tests/
├── unit/
├── integration/
└── e2e/
```

### app 目录职责

- `app/main.py`：应用入口，负责创建 FastAPI 应用并注册系统路由与业务路由。
- `app/api/v1/`：HTTP 接口层，只做参数解析、调用 service、返回响应。
- `app/clients/`：第三方客户端层，统一封装外部系统和外部模型调用。
- `app/core/`：基础设施层，放配置、日志、通用异常等横切能力。
- `app/schemas/`：接口请求体、响应体和内部数据传输对象定义。
- `app/services/`：业务编排层，负责会话、消息、对话等用例流程。
- `app/agent/`：Agent 编排层，负责状态定义、图编排、节点路由与上下文构建。
- `app/memory/`：短期记忆与检查点相关能力，不负责业务接口暴露。
- `app/knowledge/`：知识库接入层，当前阶段只计划对接 RAGFlow，不自建完整 RAG
  流水线。
- `app/tools/`：工具注册与内置工具定义，例如计算器与时间工具。
- `app/mcp/`：MCP 接入骨架，负责管理协议客户端与调用入口。
- `app/persistence/`：数据访问层，只负责数据库读写，不承担业务决策。

### tests 目录职责

- `tests/unit/`：单元测试，覆盖纯逻辑和轻量模块。
- `tests/integration/`：集成测试，覆盖 API、服务编排和模块协作。
- `tests/e2e/`：端到端测试，覆盖完整业务链路。

## 本地运行

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
uvicorn app.main:app --reload
```

服务启动后访问 `http://127.0.0.1:8000/health` 即可验证服务存活。

## 首次本地启动基础依赖

如果你的本机还没有可用的 PostgreSQL 和 Redis，最快方式是直接使用当前仓库的
`docker-compose.yml` 启动基础依赖：

```bash
copy .env.example .env
docker compose up -d postgres redis
docker compose ps
```

默认映射结果如下：

- PostgreSQL：`localhost:55432`
- Redis：`localhost:6379`

如果 `55432` 或 `6379` 仍然和你本机已有容器冲突，可以在 `.env` 中调整：

```bash
POSTGRES_HOST_PORT=65432
REDIS_HOST_PORT=6389
```

## 配置 LLM

当前单轮对话已经接入真实 LLM。运行前请在 `.env` 中至少配置：

```bash
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-4.1-mini
```

如果你使用 OpenAI 兼容接口，例如自建网关或其他兼容服务，还可以配置：

```bash
OPENAI_BASE_URL=https://your-provider.example.com/v1

```

当前后端同时提供两类聊天入口：

- 内部业务接口：`POST /api/v1/chat`
- OpenAI 兼容接口：`POST /v1/chat/completions`

其中 `/api/v1/chat` 当前采用“OpenAI 兼容请求体 + OpenAI 兼容响应体”的方式：

- 请求体使用 OpenAI Chat Completions 兼容结构
- 响应体返回 OpenAI Chat Completions 兼容结构
- 会话复用通过请求头 `X-Session-ID` 传入
- 自动创建或复用后的 `session_id` 通过响应头 `X-Session-ID` 返回

OpenAI 兼容适配层会尽量保持输入输出结构与 OpenAI Chat Completions 一致，因此后续接入 Qwen 系列或其他 OpenAI 兼容模型时，优先只需要调整兼容网关地址和模型名，而不需要修改上层调用协议。

```bash
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
OPENAI_MODEL=qwen-plus
```

设计上不在业务层硬编码供应商名称，而是透传 `model` 和 `base_url`。这样后续切换到其他 OpenAI 兼容模型时，客户端和上层调用代码可以保持稳定。
当前模型初始化和工具绑定优先复用 LangChain 官方能力：

- 使用 `langchain.chat_models.init_chat_model(...)` 初始化 OpenAI / OpenAI-compatible 模型
- 使用 `bind_tools(...)` 绑定内置工具

对于实现了 OpenAI Chat Completions 协议的模型服务，当前默认走 `model_provider="openai"` + `base_url` 的方式接入。对于带有大量非标准字段的供应商，建议优先使用 LangChain 对应的 provider-specific 集成包。

OpenAI 兼容调用示例：

```bash
curl -X POST "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-plus",
    "messages": [
      {"role": "system", "content": "你是一个简洁助手。"},
      {"role": "user", "content": "你好"}
    ]
  }'
```

当前兼容层限制：

- 当前仅支持文本消息
- 当前兼容层本身不负责会话持久化
- 当前工具能力仅开放内置 `calculator` 和 `current_datetime`

基础依赖和 LLM 配置完成后，再执行：

```bash
uvicorn app.main:app --reload
```

## Docker 运行

```bash
docker compose up --build
```

## 本地自检

```bash
ruff check .
ruff format --check .
pytest
```

## 阶段说明

当前仓库已完成工程初始化、基础设施、持久化层、基础 API 和真实 LLM 单轮对话接入。
LangGraph 多轮编排、RAGFlow 接入、工具系统和 MCP 能力会在后续阶段继续补齐。
