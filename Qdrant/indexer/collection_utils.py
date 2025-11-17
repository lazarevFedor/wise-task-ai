from __future__ import annotations

import uuid
from typing import Iterable, List
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

__all__ = [
    "_resolve_distance",
    "_normalize_point_id",
    "_normalize_ids",
    "ensure_collection",
]

def _resolve_distance(name: str) -> Distance:
    normalized = (name or "cosine").strip().lower()
    if normalized in {"cos", "cosine"}:
        return Distance.COSINE
    if normalized in {"dot", "dotproduct", "ip"}:
        return Distance.DOT
    if normalized in {"l2", "euclid", "euclidean"}:
        return Distance.EUCLID
    raise ValueError(f"Unsupported distance metric: {name}")

def _normalize_point_id(raw_id: int | str) -> int | str:
    if isinstance(raw_id, int):
        if raw_id < 0:
            raise ValueError("Point id must be non-negative integer or string")
        return raw_id
    if isinstance(raw_id, str):
        candidate = raw_id.strip()
        if not candidate:
            raise ValueError("Point id must be non-empty")
        try:
            return str(uuid.UUID(candidate))
        except ValueError:
            derived = uuid.uuid5(uuid.NAMESPACE_URL, f"qdrant-ingest::{candidate}")
            return str(derived)
    raise ValueError("Point id must be int or str")

def _normalize_ids(ids: Iterable[int | str]) -> List[int | str]:
    return [_normalize_point_id(value) for value in ids]

def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int, vector_distance: Distance) -> None:
    try:
        info = client.get_collections()
        names = [c.name for c in getattr(info, 'collections', [])]
        if collection_name in names:
            return
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=vector_distance),
        )
    except Exception:
        return
