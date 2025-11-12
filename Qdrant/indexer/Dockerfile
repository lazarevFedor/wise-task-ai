FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1 \
	HF_HOME=/root/.cache/huggingface \
	TRANSFORMERS_CACHE=/root/.cache/huggingface \
	SENTENCE_TRANSFORMERS_HOME=/root/.cache/sentence-transformers
WORKDIR /app

RUN apt-get update && apt-get install -y build-essential git gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY requirements-local.txt .
RUN pip install --no-cache-dir -r requirements-local.txt && \
	pip install --no-cache-dir -r requirements.txt

ARG EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}


COPY . .
CMD ["python", "main.py"]
