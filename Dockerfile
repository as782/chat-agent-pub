# 使用 Python 3.11 的精简镜像，减少基础镜像体积。
FROM python:3.11-slim

# 关闭 pyc 文件写入并启用无缓冲输出，方便在容器内排查问题。
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 统一容器内工作目录，避免路径分散。
WORKDIR /workspace

# 安装构建依赖，保证需要编译的 Python 包可以正常安装。
RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖描述文件，便于后续利用 Docker 层缓存。
COPY pyproject.toml README.md ./

# 再复制应用代码和测试代码，保持镜像内结构与仓库一致。
COPY app ./app
COPY tests ./tests

# 安装项目及开发依赖，便于容器内直接执行 pytest 和 ruff。
RUN python -m pip install --upgrade pip \
    && python -m pip install -e .[dev]

# 暴露 FastAPI 默认端口，供 compose 或其他容器编排工具映射。
EXPOSE 8000

# 默认启动 FastAPI 开发服务，便于本地联调。
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
