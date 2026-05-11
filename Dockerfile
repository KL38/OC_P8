# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# LightGBM links against OpenMP at runtime (libgomp.so.1). The python:3.12-slim
# base image does not ship it, so we install it explicitly. Note: packages.txt
# is ignored by HF Spaces when sdk=docker — system deps must live here.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# UV gives reproducible installs from uv.lock and is faster than pip.
COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /usr/local/bin/uv

WORKDIR /app

# Resolve runtime deps first so the layer is cached when only source code
# changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Application code + serialized artefacts. The 235 MB feature store is NOT
# bundled — it lives in a separate HF Dataset repo and is fetched at runtime
# via huggingface_hub (see api/main.py::_resolve_feature_store_path).
COPY api/ ./api/
COPY database/ ./database/
COPY models/ ./models/

# Sanity: fail fast at build time if the artefacts the API needs are missing.
RUN test -f models/model.joblib \
    && test -f models/feature_names.json \
    && test -f models/app_train_categories.json \
    && test -f models/app_train_binary_mappings.json \
    && test -f models/no_history_template.json \
    && echo "All runtime artefacts present."

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/health').status == 200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
