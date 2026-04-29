# Live Agent Proxy

这个目录提供一个可挂载到主项目 FastAPI 的透明代理服务。

当前会对外暴露以下 6 类代理能力：

- `/agent/driving`
- `/agent/event`
- `/agent/service`
- `/agent/topN`
- `/v1/chat/monitor-completions`
- `/ragflow/...`

它会把收到的 `GET` 请求按原始查询参数转发到真实上游 `http://33.69.9.33:8081`，并将上游响应体和状态码直接返回。
对于 `POST /v1/chat/monitor-completions`，它会把请求体和常用请求头透明转发到 `http://12.1.90.211:32788/v1/chat/completions`，同时保留流式和非流式两种返回方式。
对于 `/ragflow/...`，它会把任意方法和路径透明转发到真实上游 `http://33.69.3.30:8008/...`，并保留查询参数、请求体、响应头、状态码以及 SSE 流式返回。

默认情况下，这些代理接口不会启用。只有在主项目 `.env` 中设置：

```env
ENABLE_MONITOR_NETWORK_PROXY=true
```

之后，它们才会跟随主项目一起启动，并且与主项目共用同一个端口。

## 启动

启用代理后的主项目启动方式不变：

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Docker 部署时也不需要额外起第二个进程，只需要把开关打开，容器中的主 FastAPI 服务会自动带上这些接口。

## 新增环境变量

```env
ENABLE_MONITOR_NETWORK_PROXY=false
```

## 调用示例

### 1. 路线查询

```bash
curl "http://127.0.0.1:8000/agent/driving?start=杭州&end=温州"
```

### 2. 路况查询

```bash
curl "http://127.0.0.1:8000/agent/event?road=G60"
```

### 3. 服务区查询

```bash
curl "http://127.0.0.1:8000/agent/service?keyword=兰溪服务区南区"
```

### 4. 路网概览查询

```bash
curl "http://127.0.0.1:8000/agent/topN"
```

### 5. 监控网大模型非流式调用

```bash
curl "http://127.0.0.1:8000/v1/chat/monitor-completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dKyDeqHfGLEAjSUQA46bFb6d1cBe4aA4822209012a8eF925" \
  -d '{
    "model": "qwen3535ba3b",
    "messages": [{"role": "user", "content": "写一首关于春天的五言绝句"}],
    "stream": false
  }'
```

### 6. 监控网大模型流式调用

```bash
curl "http://127.0.0.1:8000/v1/chat/monitor-completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dKyDeqHfGLEAjSUQA46bFb6d1cBe4aA4822209012a8eF925" \
  -d '{
    "model": "qwen3535ba3b",
    "messages": [{"role": "user", "content": "写一首关于春天的五言绝句"}],
    "stream": true
  }'
```

### 7. RAGFlow 数据集列表

```bash
curl "http://127.0.0.1:8000/ragflow/api/v1/datasets?page=1&page_size=10" \
  -H "Authorization: Bearer ragflow-He4c0XmA3c52-O5DNg9Jup2XM0TrDO_vO_zKSfDAxzc"
```

### 8. RAGFlow 检索

```bash
curl "http://127.0.0.1:8000/ragflow/api/v1/retrieval" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ragflow-He4c0XmA3c52-O5DNg9Jup2XM0TrDO_vO_zKSfDAxzc" \
  -d '{
    "question": "绿通政策是什么",
    "dataset_ids": ["a2e9f8ff324011f198edce511781c013"],
    "top_k": 5,
    "page_size": 5
  }'
```

### 9. 本地开发接入方式

如果办公网本地运行的 `chat-agent` 需要通过该代理访问监控网 RAGFlow，可将本地开发环境中的：

```env
RAGFLOW_BASE_URL=http://办公网桥接机:22372/ragflow
```

这样本地项目仍然走原始的 RAGFlow API 路径，只是上游被这里透明转发了。

## 健康检查

```text
GET /health
```
