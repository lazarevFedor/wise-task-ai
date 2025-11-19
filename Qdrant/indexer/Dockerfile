
FROM python:3.10 AS builder
WORKDIR /build
COPY requirements-local.txt requirements-local.txt
COPY requirements.txt requirements.txt
RUN pip wheel --wheel-dir /wheels -r requirements-local.txt && \
    pip wheel --wheel-dir /wheels -r requirements.txt

FROM python:3.10-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/root/.cache/sentence-transformers
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

COPY . .
RUN chmod +x /app/auto_index.sh


ARG EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}

CMD ["python", "main.py"]
