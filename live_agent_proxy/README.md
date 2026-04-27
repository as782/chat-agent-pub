# Live Agent Proxy

这个目录提供一个可独立启动的透明代理服务，对外暴露以下 5 个接口：

- `/agent/driving`
- `/agent/event`
- `/agent/service`
- `/agent/topN`
- `/v1/chat/completions`

它会把收到的 `GET` 请求按原始查询参数转发到真实上游 `http://33.69.9.33:8081`，并将上游响应体和状态码直接返回。
对于 `POST /v1/chat/completions`，它会把请求体和常用请求头透明转发到 `http://12.1.90.211:32788/v1/chat/completions`，同时保留流式和非流式两种返回方式。

## 启动

```bash
uv run uvicorn live_agent_proxy.main:app --host 0.0.0.0 --port 8081
```

## 可选环境变量

```env
LIVE_AGENT_UPSTREAM_BASE_URL=http://33.69.9.33:8081
LLM_UPSTREAM_BASE_URL=http://12.1.90.211:32788
PROXY_TIMEOUT_SECONDS=30
PROXY_APP_HOST=0.0.0.0
PROXY_APP_PORT=8081
```

## 调用示例

### 1. 路线查询

```bash
curl "http://127.0.0.1:8081/agent/driving?start=杭州&end=温州"
```

### 2. 路况查询

```bash
curl "http://127.0.0.1:8081/agent/event?road=G60"
```

### 3. 服务区查询

```bash
curl "http://127.0.0.1:8081/agent/service?keyword=兰溪服务区南区"
```

### 4. 路网概览查询

```bash
curl "http://127.0.0.1:8081/agent/topN"
```

### 5. 大模型非流式调用

```bash
curl "http://127.0.0.1:8081/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dKyDeqHfGLEAjSUQA46bFb6d1cBe4aA4822209012a8eF925" \
  -d '{
    "model": "qwen3535ba3b",
    "messages": [{"role": "user", "content": "写一首关于春天的五言绝句"}],
    "stream": false
  }'
```

### 6. 大模型流式调用

```bash
curl "http://127.0.0.1:8081/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dKyDeqHfGLEAjSUQA46bFb6d1cBe4aA4822209012a8eF925" \
  -d '{
    "model": "qwen3535ba3b",
    "messages": [{"role": "user", "content": "写一首关于春天的五言绝句"}],
    "stream": true
  }'
```

## 健康检查

```text
GET /health
```
