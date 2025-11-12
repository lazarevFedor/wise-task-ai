FROM python:3.13 as builder

WORKDIR /app

COPY llm-service/requirements.txt .

RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.13-slim

WORKDIR /app

COPY --from=builder /root/.local /root/.local

COPY llm-service/ .

ENV PATH=/root/.local/bin:$PATH

RUN ls

CMD ["python", "app/main.py"]