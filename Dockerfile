FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /build

RUN --mount=type=cache,target=/root/.cache/pip \
    apt-get update \
    && apt-get install --yes --no-install-recommends build-essential \
    && python -m venv "$VIRTUAL_ENV" \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

RUN python - <<'PY' > requirements.txt
import tomllib
from pathlib import Path

project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
for dependency in project["project"]["dependencies"]:
    print(dependency)
PY

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install -r requirements.txt


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000

WORKDIR /workspace

RUN apt-get update \
    && apt-get install --yes --no-install-recommends curl gosu \
    && rm -rf /var/lib/apt/lists/* \
    && addgroup --system app \
    && adduser --system --ingroup app app \
    && mkdir -p /workspace \
    && chown -R app:app /workspace

COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app pyproject.toml ./pyproject.toml
COPY --chown=app:app app ./app
COPY --chown=app:app live_agent_proxy ./live_agent_proxy
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

RUN python - <<'PY'
from hashlib import sha256
from json import dumps
from pathlib import Path

workspace = Path("/workspace")
digest = sha256()
for path in sorted((workspace / "app").rglob("*")):
    if path.is_file():
        digest.update(path.relative_to(workspace).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
for path in sorted((workspace / "live_agent_proxy").rglob("*")):
    if path.is_file():
        digest.update(path.relative_to(workspace).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())

(workspace / ".build-info.json").write_text(
    dumps(
        {
            "app_source_sha256": digest.hexdigest(),
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
PY

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["sh", "-c", "exec uvicorn app.main:app --host ${APP_HOST:-0.0.0.0} --port ${APP_PORT:-8000}"]
