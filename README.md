# Chat Agent Backend

最小可用的 Agent 问答系统后端项目。

当前仓库已经完成：

- FastAPI 服务骨架
- PostgreSQL 会话与消息持久化
- Redis/内存短期状态支撑
- LangChain + LangGraph 多轮对话主链路
- RAGFlow 接入层
- 内置工具调用
- MCP 外部服务接入骨架与最小可用调用
- OpenAI 兼容输入输出接口
- pytest / Ruff / pre-commit / Docker / `uv`

## 技术栈

- Python 3.11
- FastAPI
- LangChain
- LangGraph
- PostgreSQL
- Redis
- pytest / pytest-asyncio
- Ruff
- pre-commit
- Docker / docker compose
- `uv`

## 当前能力

- `POST /api/v1/chat`
  - OpenAI Chat Completions 兼容输入输出
  - 支持同一会话多轮记忆
  - 支持知识库路由、工具调用、MCP 工具调用
- `POST /v1/chat/completions`
  - OpenAI 兼容接口
  - 复用 `/api/v1/chat` 的会话、多轮记忆、工具、知识库与流式链路
- `POST /api/v1/sessions`
  - 创建会话
- `GET /api/v1/sessions`
  - 查询会话列表
- `GET /api/v1/sessions/{session_id}`
  - 查询单个会话
- `GET /api/v1/messages/{session_id}`
  - 查询消息历史
- `POST /api/v1/knowledge/datasets/sync`
  - 同步 RAGFlow 数据集
- `POST /api/v1/knowledge/retrieval`
  - 执行知识检索
- `GET /api/v1/mcp/servers`
  - 查询 MCP 服务配置
- `POST /api/v1/mcp/servers/{server_name}/probe`
  - 探测 MCP 服务
- `GET /api/v1/mcp/servers/{server_name}/tools`
  - 列出 MCP 工具

## 目录结构

```text
app/
├── api/
│   └── v1/
├── agent/
│   └── nodes/
├── clients/
├── core/
├── knowledge/
│   └── ragflow/
├── mcp/
├── memory/
├── persistence/
├── schemas/
├── services/
├── tools/
│   └── builtin/
└── main.py
tests/
├── unit/
├── integration/
└── e2e/
docs/
└── development.md
```

目录职责：

- `app/api/v1/`：HTTP 路由层，只做参数解析、调用 service、返回响应。
- `app/services/`：业务编排层。
- `app/persistence/`：数据访问层，不做业务决策。
- `app/clients/`、`app/knowledge/ragflow/`、`app/mcp/`：第三方调用封装层。
- `app/agent/`：LangGraph 状态、图编排、节点和上下文构建。
- `app/memory/`：短期记忆、摘要、checkpoint。
- `app/tools/`：内置工具与工具注册中心。
- `tests/unit`：纯逻辑测试。
- `tests/integration`：模块协作与 API 测试。
- `tests/e2e`：端到端链路测试。

## 快速开始

1. 安装依赖

```bash
uv sync
```

2. 复制环境变量模板

```bash
copy .env.example .env
```

3. 如果本机还没有 PostgreSQL / Redis，先启动基础依赖

```bash
docker compose up -d postgres redis
```

默认宿主机端口：

- PostgreSQL：`localhost:55432`
- Redis：`localhost:6379`

4. 启动服务

```bash
uv run uvicorn app.main:app --reload
```

5. 打开文档

```text
http://127.0.0.1:8000/docs
```

## 环境变量

最小本地运行通常至少需要：

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=55432
POSTGRES_DB=chat_agent
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

REDIS_URL=redis://localhost:6379/0

RAGFLOW_BASE_URL=http://xxx:8008
RAGFLOW_API_KEY=xxxx
DEFAULT_KNOWLEDGE_DATASET_ID=xxxxx


OPENAI_API_KEY=replace-me
OPENAI_MODEL=qwen-plus
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1

PLANNER_MODEL=qwen-plus

LIVE_AGENT_BASE_URL=http://xxx:8081
LIVE_AGENT_TERMINAL_EXEC_ENABLED=false

MCP_SERVERS_JSON=[{"mcpServers":{"amap-maps-sse":{"url":"https://mcp.amap.com/sse?key=xxx"}}}]

```

开发环境如果不能直连监控网四个 live-agent 接口，可以打开
`LIVE_AGENT_TERMINAL_EXEC_ENABLED=true`，并配置：

```env
LIVE_AGENT_TERMINAL_EXEC_URL=https://frp-sea.com:28965/api/v1/mcp/terminal_exec
LIVE_AGENT_TERMINAL_TARGET_BASE_URL=http://33.69.9.33:8081
LIVE_AGENT_TERMINAL_EXEC_TIMEOUT_SECONDS=180
LIVE_AGENT_TERMINAL_EXEC_RETRIES=3
LIVE_AGENT_TERMINAL_EXEC_CURL_BINARY=curl
```

如果使用 Qwen3 一类兼容模型，项目会在非流式场景下自动补 `enable_thinking=false`，避免常见兼容网关报错。

## Thinking / Reasoning Output

For OpenAI-compatible models that support thinking mode, such as some Qwen3 / QwQ-style models:

- For non-streaming requests, send `enable_thinking: true` if you want the model to return thinking content
- Non-streaming responses now keep the thinking content inside `choices[0].message.content` only, wrapped as `<think>...</think>final answer`
- Streaming responses now keep the thinking content inside `choices[0].delta.content` only, wrapped as `<think>` tags
- `reasoning_content` is no longer exposed to the frontend as a response field
- If the upstream model returns only thinking content and no final answer, `content` may contain only `<think>...</think>`

Non-streaming request example:

```json
{
  "model": "qwen3.5-35ba3b",
  "messages": [
    {
      "role": "user",
      "content": "Please think first, then answer what 1+1 equals"
    }
  ],
  "enable_thinking": true
}
```

Non-streaming response example:

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "qwen3.5-35ba3b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "<think>First identify this as a basic addition problem, then compute 1+1.</think>1+1 equals 2."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 12,
    "total_tokens": 22
  }
}
```

Streaming SSE example:

```text
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1710000000,"model":"qwen3.5-35ba3b","choices":[{"index":0,"delta":{"role":"assistant","content":"<think>First decide it is an addition problem."},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1710000000,"model":"qwen3.5-35ba3b","choices":[{"index":0,"delta":{"role":"assistant","content":"</think>1+1 equals 2."},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1710000000,"model":"qwen3.5-35ba3b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

Notes:

- The project still patches LangChain `ChatOpenAI` so third-party OpenAI-compatible responses can keep the reasoning text internally
- If `enable_thinking: true` is not set and the model name matches `qwen3*`, non-streaming calls may still be downgraded to `enable_thinking=false`

## MCP 配置

项目当前支持外部标准 MCP 服务配置，支持：

- `http`
- `streamable_http`
- `sse`
- `stdio`

推荐 `.env` 中使用单行 JSON。

高德 `streamable_http` 示例：

```env
MCP_SERVERS_JSON={"mcpServers":{"amap-maps-streamableHTTP":{"url":"https://mcp.amap.com/mcp?key=replace-me"}}}
```

SSE 示例：

```env
MCP_SERVERS_JSON={"mcpServers":{"demo-sse":{"transport":"sse","url":"https://mcp.example.com/sse"}}}
```

stdio 示例：

```env
MCP_SERVERS_JSON=[{"name":"amap","transport":"stdio","command":"uvx","args":["amap-mcp-server"],"env":{"AMAP_MAPS_API_KEY":"replace-me"}}]
```

## 常用命令

启动服务：

```bash
uv run uvicorn app.main:app --reload
```

运行自检：

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

安装 pre-commit：

```bash
uv run pre-commit install
```

执行 pre-commit：

```bash
uv run pre-commit run --all-files
```

## 测试分层

- `tests/unit`
  - router 决策
  - context builder
  - tool registry
  - RAGFlow client 参数组装
  - repository 纯逻辑
  - 中文注释规范基线
- `tests/integration`
  - chat / sessions / messages / knowledge / mcp API
  - agent graph 主链路
- `tests/e2e`
  - 创建会话
  - 多轮发送消息
  - 调用知识库
  - 查询历史消息

## 开发规范

开发流程、Git 分支规范、Commit 规范、PR 规范、中文注释规范见：

- [docs/development.md](/c:/Users/wengkaibin/DATA/WorkSpace/Test spaces/chat-agent/docs/development.md)

## 日志

- 应用日志同时输出到控制台和文件。
- Docker 部署默认把宿主机 `./logs` 挂载到容器内 `/workspace/logs`。
- 容器启动时会先自动修复日志目录权限，再降权启动应用，减少 Linux 绑定挂载目录的权限问题。
- 默认日志文件为 `./logs/chat-agent.log`，按天轮转，默认保留 14 份历史日志。
- 可通过 `LOG_TO_FILE`、`LOG_DIR`、`LOG_FILE_NAME`、`LOG_ROTATE_WHEN`、`LOG_ROTATE_INTERVAL`、`LOG_BACKUP_COUNT` 调整策略。
- 如果文件日志暂时不可写，应用会自动降级为仅输出控制台日志，避免服务启动失败。
- 实时查看容器日志可使用 `docker compose logs -f app`。
- 直接查看持久化日志文件可使用 `logs/chat-agent.log` 及轮转后的 `logs/chat-agent.log.YYYY-MM-DD`。
