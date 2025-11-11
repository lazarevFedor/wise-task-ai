FROM python:3.13 as builder

WORKDIR /llm-service

COPY requirements.txt .

RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.13-slim

WORKDIR /llm-service

COPY --from=builder /root/.local /root/.local

COPY . .

ENV PATH=/root/.local/bin:$PATH

CMD ["python", "app/main.py"]