#!/usr/bin/env python3

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional
import requests

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointIdsList, PointStruct


from collection_utils import (
    _resolve_distance,
    _normalize_point_id,
    _normalize_ids,
    ensure_collection as ensure_collection_util,
)


QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
from config_utils import DEFAULT_COLLECTION, VECTOR_SIZE

VECTOR_DISTANCE = _resolve_distance(os.getenv("VECTOR_DISTANCE", "cosine"))
from embedding_utils import EMBEDDING_MODEL, _get_embedding_model

_client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
    prefer_grpc=False,
    grpc_port=QDRANT_GRPC_PORT,
)


from schemas import (
    UpsertRequest,
    DeleteRequest,
    EmbedRequest,
    SearchRequest,
    RagRequest,
    AutoRequest,
)
from search_rag import run_search_pipeline, run_rag_pipeline


def ensure_collection(collection_name: str) -> None:
    ensure_collection_util(_client, collection_name, VECTOR_SIZE, VECTOR_DISTANCE)


app = FastAPI(title="Embedding Ingest API", default_response_class=ORJSONResponse)


@app.on_event("startup")
def _startup() -> None:
    ensure_collection(DEFAULT_COLLECTION)
    try:
        threading.Thread(target=_get_embedding_model, daemon=True).start()
    except Exception:
        pass


API_KEY = os.getenv("API_KEY")


@app.middleware("http")
async def _auth_middleware(request: Request, call_next):
    if API_KEY:
        provided = request.headers.get("x-api-key") or request.headers.get("X-Api-Key")
        if provided != API_KEY:
            return ORJSONResponse({"error": "unauthorized"}, status_code=401)
    return await call_next(request)


@app.get("/health")
def health() -> Dict[str, Any]:
    points_count = None
    try:
        r = requests.get(
            f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{DEFAULT_COLLECTION}",
            timeout=3,
        )
        if r.ok:
            js = r.json()
            points_count = js.get("result", {}).get("points_count")
    except Exception:
        points_count = None
    return {
        "status": "ok",
        "collection": DEFAULT_COLLECTION,
        "points_count": points_count,
        "vector_size": VECTOR_SIZE,
        "distance": VECTOR_DISTANCE.value,
        "model": EMBEDDING_MODEL,
        "query_prefix": "",
    }


@app.post("/v1/embed")
def embed_texts(body: EmbedRequest) -> Dict[str, Any]:
    model = _get_embedding_model()
    if hasattr(model, "embed"):
        vectors = list(model.embed(body.texts))
    else:
        vectors = model.encode(
            body.texts, show_progress_bar=False, convert_to_numpy=True
        )
    try:
        import numpy as np

        if isinstance(vectors, np.ndarray):
            vectors_list = [v.tolist() for v in vectors]
        else:
            vectors_list = [list(map(float, v)) for v in vectors]
    except Exception:
        vectors_list = [list(map(float, v)) for v in vectors]
    size = len(vectors_list[0])
    if VECTOR_SIZE and size != VECTOR_SIZE:
        raise HTTPException(
            status_code=500,
            detail=f"Embedding dimension {size} != configured {VECTOR_SIZE}",
        )
    return {"model": EMBEDDING_MODEL, "vector_size": size, "vectors": vectors_list}


@app.post("/v1/upsert")
def upsert_embeddings(body: UpsertRequest) -> Dict[str, Any]:
    collection = body.collection or DEFAULT_COLLECTION
    ensure_collection(collection)

    points: List[PointStruct] = []
    stored_ids: List[int | str] = []
    for entry in body.points:
        try:
            normalized_id = _normalize_point_id(entry.id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        payload = dict(entry.payload or {})
        if isinstance(entry.id, str):
            canonical = str(normalized_id)
            if canonical != entry.id and "external_id" not in payload:
                payload["external_id"] = entry.id

        points.append(
            PointStruct(
                id=normalized_id,
                vector=entry.vector,
                payload=payload,
            )
        )
        stored_ids.append(normalized_id)

    operation_info = _client.upsert(
        collection_name=collection,
        points=points,
        wait=body.wait,
    )
    return {
        "status": getattr(operation_info, "status", "acknowledged"),
        "collection": collection,
        "upserted": len(points),
        "operation_id": getattr(operation_info, "operation_id", None),
        "time": getattr(operation_info, "time", None),
        "stored_ids": stored_ids,
    }


@app.post("/v1/delete")
def delete_points(body: DeleteRequest) -> Dict[str, Any]:
    collection = body.collection or DEFAULT_COLLECTION
    ensure_collection(collection)
    try:
        normalized_ids = _normalize_ids(body.ids)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _client.delete(
        collection_name=collection,
        points_selector=PointIdsList(points=normalized_ids),
    )
    return {
        "status": "ok",
        "deleted_ids": body.ids,
        "collection": collection,
    }


@app.post("/v1/ensure-collection")
def ensure_collection_endpoint(collection: str = DEFAULT_COLLECTION) -> Dict[str, Any]:
    ensure_collection(collection)
    return {
        "status": "ok",
        "collection": collection,
        "vector_size": VECTOR_SIZE,
        "distance": VECTOR_DISTANCE.value,
    }


@app.get("/v1/collection-info")
def collection_info(collection: str = DEFAULT_COLLECTION) -> Dict[str, Any]:
    try:
        r = requests.get(
            f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{collection}", timeout=5
        )
        r.raise_for_status()
        js = r.json()
        result = js.get("result", {})
        return {
            "collection": collection,
            "status": js.get("status", "ok"),
            "points_count": result.get("points_count"),
            "indexed_vectors_count": result.get("indexed_vectors_count"),
            "segments_count": result.get("segments_count"),
        }
    except Exception as e:
        return {
            "collection": collection,
            "error": str(e),
            "status": "error",
        }


@app.post("/v1/search")
def search_by_text(body: SearchRequest) -> Dict[str, Any]:
    """Search by text query - embedding computed on server side."""
    try:
        return run_search_pipeline(body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/v1/search-simple")
def search_simple(
    q: str, limit: int = 5, collection: Optional[str] = None, max_chars: int = 800
) -> Dict[str, Any]:
    req = SearchRequest(
        query=q, limit=limit, collection=collection, max_chars=max_chars
    )
    return search_by_text(req)


@app.post("/v1/rag")
def rag_context(body: RagRequest) -> Dict[str, Any]:
    """Return compact RAG-ready context string and structured citations."""
    try:
        return run_rag_pipeline(body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/v1/rag-simple")
def rag_simple(
    q: str, limit: int = 5, context_chars: int = 2000, collection: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience GET endpoint: /v1/rag-simple?q=...&limit=5&context_chars=2000"""
    req = RagRequest(
        query=q, limit=limit, context_chars=context_chars, collection=collection
    )
    return rag_context(req)


@app.post("/v1/auto")
def auto_dispatch(body: AutoRequest) -> Dict[str, Any]:
    collection = body.collection or DEFAULT_COLLECTION

    if body.points:
        req = UpsertRequest(collection=collection, points=body.points, wait=body.wait)
        res = upsert_embeddings(req)
        return {"action": "upsert", **res}

    if body.delete_ids:
        req = DeleteRequest(collection=collection, ids=body.delete_ids)
        res = delete_points(req)
        return {"action": "delete", **res}

    if body.texts:
        req = EmbedRequest(texts=body.texts)
        res = embed_texts(req)
        return {"action": "embed", **res}

    if isinstance(body.query, str) and body.query.strip():
        req = RagRequest(
            query=body.query.strip(),
            collection=collection,
            limit=body.limit,
            context_chars=body.context_chars,
            include_sources=body.include_sources,
            with_chunk_text=body.with_chunk_text,
            include_debug=body.include_debug,
            min_term_hits=body.min_term_hits,
            diversify=body.diversify,
            mmr_lambda=body.mmr_lambda,
            internal_candidates=body.internal_candidates,
        )
        res = rag_context(req)
        return {"action": "rag", **res}

    raise HTTPException(
        status_code=422, detail="Provide one of: points, delete_ids, texts, or query"
    )
