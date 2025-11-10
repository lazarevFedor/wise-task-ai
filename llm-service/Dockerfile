FROM python:3.13

WORKDIR /llm-service

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir

COPY . .

CMD ["python", "app/main.py", "localhost:8081"]
