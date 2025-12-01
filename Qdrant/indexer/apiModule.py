import os
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

from searchModule import Searcher


QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "latex_books")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

API_KEY = os.getenv("API_KEY")

DEFAULT_SEARCH_LIMIT = int(os.getenv("DEFAULT_SEARCH_LIMIT", "5"))
DEFAULT_RAG_LIMIT = int(os.getenv("DEFAULT_RAG_LIMIT", "5"))
DEFAULT_CONTEXT_CHARS = int(os.getenv("DEFAULT_CONTEXT_CHARS", "2000"))
DEFAULT_SCORE_THRESHOLD = float(os.getenv("DEFAULT_SCORE_THRESHOLD", "0.3"))


class SearchRequest(BaseModel):
    query: str
    limit: int = DEFAULT_SEARCH_LIMIT
    score_threshold: float = DEFAULT_SCORE_THRESHOLD


class SearchResponse(BaseModel):
    query: str
    results: List[Dict]
    count: int


class HealthResponse(BaseModel):
    status: str
    collection: str
    points_count: Optional[int] = None


app = FastAPI(
    title="RAG Search API",
    description="API for qdrant-db",
    version="2.0.0",
    default_response_class=ORJSONResponse,
)

searcher: Optional[Searcher] = None


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if API_KEY:
        if request.url.path in ["/", "/health"]:
            return await call_next(request)
        provided_key = request.headers.get("X-Api-Key")
        if provided_key != API_KEY:
            return ORJSONResponse(
                {"error": "Invalid or missing API key"},
                status_code=401
            )
    return await call_next(request)


@app.on_event("startup")
def startup_event():
    global searcher

    logger.info("Запуск API сервиса...")
    logger.info(f"Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
    logger.info(f"Коллекция: {COLLECTION_NAME}")
    logger.info(f"Модель: {EMBEDDING_MODEL}")
    logger.info(f"API Key: {'настроен ' if API_KEY else 'не задан (публичный доступ)'}")
    logger.info(f"limits:search={DEFAULT_SEARCH_LIMIT}"
          f",rag={DEFAULT_RAG_LIMIT},"
          f"ctx={DEFAULT_CONTEXT_CHARS}")

    try:
        searcher = Searcher(
            qdrant_host=QDRANT_HOST,
            qdrant_port=QDRANT_PORT,
            collection_name=COLLECTION_NAME,
            embedding_model=EMBEDDING_MODEL,
        )
        logger.info("Поисковик инициализирован")
    except Exception as e:
        logger.error(f"Ошибка инициализации: {e}")
        raise


@app.get("/health")
def health_check():
    if searcher is None:
        raise HTTPException(status_code=503, detail="Searcher not initialized")

    try:
        info = searcher.client.get_collection(COLLECTION_NAME)
        return {
            "status": "ok",
            "collection": COLLECTION_NAME,
            "points_count": info.points_count,
            "vector_size": 384,
            "distance": "cosine",
            "model": EMBEDDING_MODEL,
            "query_prefix": "",
            "config": {
                "default_search_limit": DEFAULT_SEARCH_LIMIT,
                "default_rag_limit": DEFAULT_RAG_LIMIT,
                "default_context_chars": DEFAULT_CONTEXT_CHARS,
                "default_score_threshold": DEFAULT_SCORE_THRESHOLD,
                "api_key_enabled": API_KEY is not None,
            },
        }
    except Exception:
        return {
            "status": "degraded",
            "collection": COLLECTION_NAME,
            "points_count": None,
            "vector_size": 384,
            "distance": "cosine",
            "model": EMBEDDING_MODEL,
            "query_prefix": "",
            "config": {
                "default_search_limit": DEFAULT_SEARCH_LIMIT,
                "default_rag_limit": DEFAULT_RAG_LIMIT,
                "default_context_chars": DEFAULT_CONTEXT_CHARS,
                "default_score_threshold": DEFAULT_SCORE_THRESHOLD,
                "api_key_enabled": API_KEY is not None,
            },
        }


@app.post("/v1/search")
def search(request: SearchRequest):
    if searcher is None:
        raise HTTPException(status_code=503, detail="Searcher not initialized")

    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        raw_results = searcher.search(
            query=request.query,
            limit=request.limit,
            score_threshold=request.score_threshold,
        )

        results = [
            {
                "id": r.get("id"),
                "score": r.get("final_score"),
                "payload": {
                    "title": r.get("title", ""),
                    "source": r.get("source", ""),
                    "chunk_index": r.get("chunk_index", 0),
                    "text": r.get("text", "")
                }

            }
            for r in raw_results
        ]

        return {
            "query": request.query,
            "collection": COLLECTION_NAME,
            "results": results,
            "count": len(results),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.get("/v1/rag")
def rag(
    q: str,
        limit: int = DEFAULT_RAG_LIMIT,
        context_chars: int = DEFAULT_CONTEXT_CHARS,
        collection: Optional[str] = None
):
    if searcher is None:
        raise HTTPException(status_code=503, detail="Searcher not initialized")

    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query 'q' cannot be empty")

    try:
        raw_results = searcher.search(query=q, limit=limit)

        chunks = []
        context_parts = []
        used = 0

        for i, result in enumerate(raw_results, 1):
            title = result.get("title", "")
            source = result.get("source", "")
            text = result.get("text", "")
            chunk_index = result.get("chunk_index", 0)

            if context_chars and used + len(text) > context_chars:
                remaining = context_chars - used
                if remaining > 100:
                    text = text[:remaining] + "..."
                else:
                    break

            header = f"[{i}] {title} ({source}, chunk {chunk_index})"
            context_parts.append(header + "\n" + text)

            chunks.append(
                {
                    "rank": i,
                    "id": result.get("id"),
                    "score": result.get("final_score"),
                    "title": title,
                    "source": source,
                    "chunk_index": chunk_index,
                    "text": text,
                }
            )

            used += len(text)
            if context_chars and used >= context_chars:
                break

        context = "\n\n".join(context_parts)

        return {
            "query": q,
            "collection": collection or COLLECTION_NAME,
            "context": context,
            "chunks": chunks,
            "count": len(chunks),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}")


@app.get("/")
def root():
    return {
        "name": "RAG Search API",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "search": "/search (POST)",
            "search_simple": "/search-simple?q=query&limit=5",
            "rag": "/rag?q=query&limit=5",
        },
        "collection": COLLECTION_NAME,
        "model": EMBEDDING_MODEL,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
