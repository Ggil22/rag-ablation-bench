# API image. PyTorch wheels bundle the CUDA runtime, so a slim Python base is enough; the host
# needs only the NVIDIA driver and the NVIDIA Container Toolkit (`docker run --gpus all`).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/models \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, so code changes do not reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev --extra models --extra serve

COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-dev --extra models --extra serve

# Build the corpus into the image: it is small (a few MB) and pinned by `as_of`.
RUN uv run --no-sync rab corpus boe

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"
CMD ["uv", "run", "--no-sync", "uvicorn", "rag_ablation_bench.serve.app:app", \
     "--host", "0.0.0.0", "--port", "8080"]
