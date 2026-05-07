# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# UV gives reproducible installs from uv.lock and is faster than pip.
COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /usr/local/bin/uv

WORKDIR /app

# Resolve runtime deps first so the layer is cached when only source code
# changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Application code + serialized artefacts.
COPY api/ ./api/
COPY models/ ./models/
COPY data/ ./data/

# Sanity: fail fast at build time if the artefacts the API needs are missing.
RUN test -f models/model.joblib \
    && test -f models/feature_names.json \
    && test -f models/app_train_categories.json \
    && test -f models/app_train_binary_mappings.json \
    && test -f models/no_history_template.json \
    && test -f data/features_store.parquet \
    && echo "All runtime artefacts present."

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/health').status == 200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
