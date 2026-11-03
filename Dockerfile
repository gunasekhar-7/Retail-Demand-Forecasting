FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    HOME=/home/appuser \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /uvx /bin/

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app
RUN chown appuser:appuser /app
USER appuser

COPY --chown=appuser:appuser pyproject.toml uv.lock README.md ./

# Dashboard image: core runtime plus Streamlit. No Prophet or DuckDB.
FROM base AS dashboard

RUN uv sync --frozen --no-dev --no-install-project --extra dashboard
COPY --chown=appuser:appuser src ./src
RUN uv sync --frozen --no-dev --extra dashboard

EXPOSE 8501
CMD ["streamlit", "run", "src/demand/dashboard/app.py", \
     "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"]

# Production API image: base runtime dependencies only. Last stage is the default.
FROM base AS api

RUN uv sync --frozen --no-dev --no-install-project
COPY --chown=appuser:appuser src ./src
RUN uv sync --frozen --no-dev

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health/ready >/dev/null || exit 1

CMD ["uvicorn", "demand.service.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
