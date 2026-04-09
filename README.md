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

MCP_SERVERS_JSON=[{"mcpServers":{"amap-maps-sse":{"url":"https://mcp.amap.com/sse?key=xxx"}}}]

```

如果使用 Qwen3 一类兼容模型，项目会在非流式场景下自动补 `enable_thinking=false`，避免常见兼容网关报错。

## Thinking / Reasoning Output

对于支持 thinking 模式的 OpenAI 兼容模型，例如部分 Qwen3 / QwQ 风格模型：

- 非流式请求如果希望返回思考内容，请显式传 `enable_thinking: true`
- 非流式响应会在 `choices[0].message.reasoning_content` 返回思考内容
- 非流式响应还会把思考内容额外拼进 `choices[0].message.content`，格式为 `<think>...</think>最终答案`
- 流式响应会在 SSE 的 `choices[0].delta.reasoning_content` 返回思考增量
- 流式响应也会在 `choices[0].delta.content` 中同步输出可直接渲染的 `<think>` 包裹内容
- 如果上游模型只返回思考内容、不返回最终答案，则 `content` 可能为空，但 `reasoning_content` 仍可能有值

非流式请求示例：

```json
{
  "model": "qwen3.5-35ba3b",
  "messages": [
    {
      "role": "user",
      "content": "请先思考，再回答 1+1 等于几"
    }
  ],
  "enable_thinking": true
}
```

非流式响应示例：

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
        "content": "<think>先识别这是一个基础加法问题，然后直接计算 1+1。</think>1+1 等于 2。",
        "reasoning_content": "先识别这是一个基础加法问题，然后直接计算 1+1。"
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

流式 SSE 片段示例：

```text
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1710000000,"model":"qwen3.5-35ba3b","choices":[{"index":0,"delta":{"role":"assistant","content":"<think>先判断题目是加法。","reasoning_content":"先判断题目是加法。"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1710000000,"model":"qwen3.5-35ba3b","choices":[{"index":0,"delta":{"role":"assistant","content":"</think>1+1 等于 2。"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1710000000,"model":"qwen3.5-35ba3b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

说明：

- 当前项目已经对 LangChain 的 `ChatOpenAI` 做了兼容补丁，用于保留第三方 OpenAI 兼容接口返回的 `reasoning_content`
- 如果未显式传 `enable_thinking: true`，且模型名匹配 `qwen3*`，非流式调用仍可能被自动降级为 `enable_thinking=false`

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
