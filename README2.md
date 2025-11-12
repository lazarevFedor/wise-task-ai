# Qdrant LaTeX Search + RAG

Поисковый сервис по LaTeX-источникам на Qdrant с удобным HTTP API. Умеет:
- индексировать файлы `.tex` в чанки и строить эмбеддинги; 
- гибридный поиск `/v1/search` (векторный + лексика + эвристики + “сшивание” соседних чанков);
- собирать длинный контекст для ответов `/v1/rag` (RAG: Retrieval-Augmented Generation);
- принимать готовые эмбеддинги по `/v1/upsert` (для внешних сервисов, например Go).

Сервис в docker-compose и может быть доступен в той же сети Compose (DNS‑имя `ingest-api:8080`).

## Состав
- qdrant — векторное хранилище (порты 6333/6334).
- indexer — индексатор LaTeX: чанкинг, эмбеддинги, загрузка в Qdrant.
- ingest-api — FastAPI на 8080: /health, /v1/search, /v1/rag, /v1/upsert, /v1/delete, /v1/ensure-collection, /v1/collection-info, /v1/auto.

## Требования
- Docker + Docker Compose
- Windows PowerShell команды приведены ниже (Linux/macOS эквивалентны)

## Быстрый старт (локально)
1) Скопируйте конфиг
```powershell
Copy-Item .env.example .env -Force
```
При желании включите защиту API ключом: задайте `API_KEY=...` в `.env`.

2) Поднимите хранилище и API
```powershell
docker compose up -d qdrant ingest-api
```

3) Первая индексация (один раз)
```powershell
docker compose run --rm indexer python main.py --recreate --batch-size 64
```

4) Проверка
```powershell
Invoke-RestMethod http://localhost:8080/health
```

Повторные перезапуски сервиса не требуют переиндексации — данные сохраняются в volume.

## Как использовать API
Базовый URL локально: `http://localhost:8080` (внутри Compose: `http://ingest-api:8080`).

Основные эндпоинты:
- GET /health — состояние и параметры коллекции
- POST /v1/search — гибридный поиск (возвращает топ‑результаты + сшивание соседей)
- POST /v1/rag — один длинный контекст + список цитат (готово для LLM)
- POST /v1/upsert — загрузить векторы (если строите эмбеддинги вне сервиса)
- POST /v1/delete — удалить точки по id
- POST /v1/ensure-collection — создать коллекцию при необходимости
- GET /v1/collection-info — краткая информация о коллекции
- POST /v1/auto — “умный” маршрутизатор (upsert/delete/embed/rag) по одному запросу

Примеры (опционально):
```powershell
curl -X POST http://localhost:8080/v1/search -H "Content-Type: application/json" -d '{"query":"алгоритм дейкстры","limit":5}'

curl -X POST http://localhost:8080/v1/rag -H "Content-Type: application/json" -d '{"query":"свойства барицентра дерева"}'
```

Если включён API ключ (`API_KEY`), передавайте заголовок `X-API-Key: <ключ>`.

## Интеграция с Go в одном docker-compose
В том же compose ваш Go‑сервис может вызывать API по адресу `http://ingest-api:8080`.

Пример фрагмента `docker-compose.yml`:
```yaml
services:
  go-app:
    build: ./go-app
    depends_on:
      - ingest-api
    environment:
      - INGEST_API_BASE=http://ingest-api:8080
      - API_KEY=${API_KEY} # если включена защита API
    # network: общий с ingest-api (по умолчанию default)
```

Мини‑пример кода на Go:
```go
reqBody := bytes.NewBuffer([]byte(`{"query":"алгоритм дейкстры","limit":5}`))
req, _ := http.NewRequest("POST", os.Getenv("INGEST_API_BASE")+"/v1/search", reqBody)
req.Header.Set("Content-Type", "application/json")
if key := os.Getenv("API_KEY"); key != "" { req.Header.Set("X-API-Key", key) }
resp, err := http.DefaultClient.Do(req)
```

Если Go‑сервис в другом compose, создайте общую внешнюю сеть:
```yaml
# файл A: сеть
networks:
  qnet:
    external: true

# файл B (Qdrant + ingest-api)
services:
  ingest-api:
    networks: [qnet]
  qdrant:
    networks: [qnet]

# файл C (Go)
services:
  go-app:
    networks: [qnet]
    environment:
      - INGEST_API_BASE=http://ingest-api:8080
```

## Интеграция в внешний compose (wise-task-ai)

Если у вас есть отдельный проект с docker-compose (например, `wise-task-ai`), добавьте сервис API в их compose и укажите `QDRANT_HOST` равным имени их контейнера Qdrant (например, `qdrant_db`). Готовый фрагмент есть в `examples/docker-compose.wise-task-ai.yml`.

Кратко (фрагмент для включения):

```yaml
services:
  qdrant_ingest:
    container_name: qdrant_ingest
    build:
      # скорректируйте путь к этому репозиторию относительно файла compose
      context: ../Qdrant/indexer
    depends_on:
      - qdrant_db
    ports:
      - "8081:8080"   # внешний порт 8081 -> внутренний 8080 (uvicorn)
    env_file:
      - ../.env
    environment:
      - QDRANT_HOST=qdrant_db
      - QDRANT_PORT=6333
      - QDRANT_GRPC_PORT=6334
      # - COLLECTION_NAME=latex_books
      # - EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
      # - VECTOR_SIZE=384
    networks:
      - core_network
    command: ["uvicorn", "ingest_service:app", "--host", "0.0.0.0", "--port", "8080"]

networks:
  core_network:
    driver: bridge
```

После добавления сервис будет доступен по `http://localhost:8081` (или по внутреннему адресу `http://qdrant_ingest:8080` для других контейнеров в той же сети). Убедитесь, что в среде указан `API_KEY`, если включена защита.

## Конфигурация (через .env)
Смотрите полный список и комментарии в `.env.example`. Ключевые:
- CHUNK_MIN_LEN / CHUNK_MAX_LEN / CHUNK_OVERLAP — размер и перекрытие чанков (для indexer). При изменении — переиндексация.
- SEARCH_* — дефолты поисковой выдачи и сшивания соседей.
- RAG_* — дефолты сборки длинного контекста.
- WEIGHT_* / TH_* — тонкая настройка гибридного ранжирования.
- API_KEY — опциональная защита API через заголовок `X-API-Key`.

Пример настроек для длинных ответов без переиндексации:
```env
SEARCH_MAX_CHARS=5000
SEARCH_STITCH_NEIGHBORS=true
SEARCH_STITCH_AFTER=8
RAG_CONTEXT_CHARS=5000
RAG_STITCH_AFTER=6
```

## Индексация и обновления
- Первая загрузка: `docker compose run --rm indexer python main.py --recreate --batch-size 64`
- Добавление новых файлов: `docker compose run --rm indexer python main.py --batch-size 64`
- Полная переиндексация (меняли CHUNK_* или модель): `--recreate`

## Мониторинг и эксплуатация
```powershell
docker compose logs -f ingest-api
docker compose logs -f qdrant
Invoke-RestMethod http://localhost:8080/health
```

## Производственные заметки
- Модель по умолчанию: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384‑d) — быстрая и дешёвая.
- Хранение данных — Docker volume `qdrant_storage` (объявлен в compose).
- Безопасность: включите `API_KEY` и прокидывайте заголовок; не логируйте ключ в access‑логах.

## Полезные утилиты
- `test_api.py` — локально проверяет /health, /v1/search и /v1/rag. Поддерживает `API_KEY` из окружения и флаги `--rag`, `--limit`, `--context-chars`.

## FAQ
— Изменил .env, но эффекта нет? Перезапустите контейнер API: `docker compose restart ingest-api`.
— Хочу «более цельные» ответы: увеличьте CHUNK_MAX_LEN (+переиндексация) или расширьте SEARCH/RAG_STITCH_* и *_MAX_CHARS.
— Модель другая? Обновите EMBEDDING_MODEL и VECTOR_SIZE на обоих сервисах и выполните полную переиндексацию.

