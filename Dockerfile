# Pipeline image (feature pipeline by default, training via --entrypoint): pinned Python, pinned uv, locked dependencies, non-root.
FROM python:3.14-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# dependencies first, so code changes do not invalidate this layer
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project
COPY README.md ./
COPY src ./src
COPY configs ./configs
RUN uv sync --locked --no-dev

FROM python:3.14-slim
# LightGBM needs the OpenMP runtime, which the slim image does not ship.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN groupadd --gid 1001 app && useradd --uid 1001 --gid app --no-create-home app
WORKDIR /app
COPY --from=builder --chown=app:app /app /app
ENV PATH="/app/.venv/bin:$PATH" DATA_DIR=/app/data PYTHONUNBUFFERED=1
RUN mkdir -p /app/data && chown app:app /app/data
USER app
ENTRYPOINT ["python", "-m", "nectarflow.features.pipeline"]
