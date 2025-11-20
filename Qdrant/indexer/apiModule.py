import os
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel

from searchModule import Searcher


class SearchRequest(BaseModel):
    query: str
    limit: int = 5
    score_threshold: float = 0.3


class SearchResponse(BaseModel):
    query: str
    results: List[Dict]
    count: int


class HealthResponse(BaseModel):
    status: str
    collection: str
    points_count: Optional[int] = None


QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "latex_books")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

app = FastAPI(
    title="RAG Search API",
    description="API for qdrant-db",
    version="2.0.0",
    default_response_class=ORJSONResponse,
)

searcher: Optional[Searcher] = None


@app.on_event("startup")
def startup_event():
    global searcher

    print("Запуск API сервиса...")
    print(f"  Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"  Коллекция: {COLLECTION_NAME}")
    print(f"  Модель: {EMBEDDING_MODEL}")

    try:
        searcher = Searcher(
            qdrant_host=QDRANT_HOST,
            qdrant_port=QDRANT_PORT,
            collection_name=COLLECTION_NAME,
            embedding_model=EMBEDDING_MODEL,
        )
        print("Поисковик инициализирован")
    except Exception as e:
        print(f"Ошибка инициализации: {e}")
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
                    "text": r.get("text", ""),
                },
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


@app.get("/v1/search-simple")
def search_simple(
    q: str, limit: int = 5, collection: Optional[str] = None, max_chars: int = 800
):
    if searcher is None:
        raise HTTPException(status_code=503, detail="Searcher not initialized")

    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query 'q' cannot be empty")

    try:
        raw_results = searcher.search(query=q, limit=limit, score_threshold=0.3)

        results = [
            {
                "id": r.get("id"),
                "score": r.get("final_score"),
                "payload": {
                    "title": r.get("title", ""),
                    "source": r.get("source", ""),
                    "chunk_index": r.get("chunk_index", 0),
                    "text": r.get("text", ""),
                },
            }
            for r in raw_results
        ]

        return {
            "query": q,
            "collection": collection or COLLECTION_NAME,
            "results": results,
            "count": len(results),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.get("/v1/rag-simple")
def rag_simple(
    q: str, limit: int = 5, context_chars: int = 2000, collection: Optional[str] = None
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
