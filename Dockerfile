FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    LLM_TOPOLOGY_BOOTSTRAPPED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        chromium \
        fonts-dejavu-core \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock .python-version README.md LICENSE ./
COPY src ./src
COPY app.py ./
COPY scripts ./scripts
COPY tests ./tests

RUN uv sync --frozen --extra dev
ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8000
CMD ["python", "app.py", "--no-browser", "--host", "0.0.0.0", "--port", "8000"]
