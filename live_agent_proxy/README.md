# Live Agent Proxy

这个目录提供一个可挂载到主项目 FastAPI 的透明代理服务，对外暴露以下 5 个接口：

- `/agent/driving`
- `/agent/event`
- `/agent/service`
- `/agent/topN`
- `/v1/chat/monitor-completions`

它会把收到的 `GET` 请求按原始查询参数转发到真实上游 `http://33.69.9.33:8081`，并将上游响应体和状态码直接返回。
对于 `POST /v1/chat/monitor-completions`，它会把请求体和常用请求头透明转发到 `http://12.1.90.211:32788/v1/chat/completions`，同时保留流式和非流式两种返回方式。

默认情况下，这 5 个接口不会启用。只有在主项目 `.env` 中设置：

```env
ENABLE_MONITOR_NETWORK_PROXY=true
```

之后，它们才会跟随主项目一起启动，并且与主项目共用同一个端口。

## 启动

启用代理后的主项目启动方式不变：

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Docker 部署时也不需要额外起第二个进程，只需要把开关打开，容器中的主 FastAPI 服务会自动带上这 5 个接口。

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

## 健康检查

```text
GET /health
```
