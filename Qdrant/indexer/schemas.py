from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, root_validator, validator

from config_utils import (
    _getenv_bool,
    DEFAULT_SEARCH_MAX_CHARS,
    DEFAULT_SEARCH_WITH_CHUNK_TEXT,
    DEFAULT_SEARCH_INCLUDE_DEBUG,
    DEFAULT_SEARCH_STITCH_NEIGHBORS,
    DEFAULT_SEARCH_STITCH_BEFORE,
    DEFAULT_SEARCH_STITCH_AFTER,
    DEFAULT_SEARCH_STITCH_USE_CHUNK_TEXT,
    DEFAULT_SEARCH_LIMIT,
    VECTOR_SIZE,
)


class PointIn(BaseModel):
    id: int | str
    vector: List[float] = Field(..., min_items=1)
    payload: Optional[Dict[str, Any]] = None

    @validator("vector")
    def _validate_vector(cls, value: List[float]) -> List[float]:
        if VECTOR_SIZE > 0 and len(value) != VECTOR_SIZE:
            raise ValueError(f"Vector size {len(value)} != expected {VECTOR_SIZE}")
        return value


class UpsertRequest(BaseModel):
    collection: Optional[str] = None
    points: List[PointIn] = Field(..., min_items=1)
    wait: bool = False


class DeleteRequest(BaseModel):
    collection: Optional[str] = None
    ids: List[int | str] = Field(..., min_items=1)


class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1)

    @root_validator(pre=True)
    def _coerce_list(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "texts" in values:
            data = values["texts"]
            if isinstance(data, str):
                values["texts"] = [data]
            return values
        text_value = values.get("text")
        if isinstance(text_value, str):
            values["texts"] = [text_value]
            return values
        raise ValueError("Provide 'texts' (list) or 'text' (string)")

    @validator("texts")
    def _non_empty(cls, value: List[str]) -> List[str]:
        cleaned = [t.strip() for t in value if t.strip()]
        if not cleaned:
            raise ValueError("texts must contain at least one non-empty string")
        return cleaned


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    collection: Optional[str] = None
    limit: int = Field(default=DEFAULT_SEARCH_LIMIT, ge=1, le=100)
    max_chars: Optional[int] = Field(
        default=DEFAULT_SEARCH_MAX_CHARS,
        ge=0,
        description="Ограничить размер возвращаемого текста",
    )
    with_chunk_text: bool = Field(
        default=DEFAULT_SEARCH_WITH_CHUNK_TEXT,
        description="Возвращать исходный chunk_text вместо display window",
    )
    include_debug: bool = Field(
        default=DEFAULT_SEARCH_INCLUDE_DEBUG,
        description="Включить диагностические поля скоринга в ответе",
    )
    stitch_neighbors: bool = Field(
        default=DEFAULT_SEARCH_STITCH_NEIGHBORS,
        description="Сшивать соседние чанки вокруг результата",
    )
    stitch_before: int = Field(
        default=DEFAULT_SEARCH_STITCH_BEFORE,
        ge=0,
        le=10,
        description="Сколько чанков взять до найденного",
    )
    stitch_after: int = Field(
        default=DEFAULT_SEARCH_STITCH_AFTER,
        ge=0,
        le=20,
        description="Сколько чанков взять после найденного",
    )
    stitch_use_chunk_text: bool = Field(
        default=DEFAULT_SEARCH_STITCH_USE_CHUNK_TEXT,
        description="При сшивании использовать chunk_text вместо text",
    )


class RagRequest(BaseModel):
    query: str = Field(..., min_length=1)
    collection: Optional[str] = None
    limit: int = Field(default=int(os.getenv("RAG_DEFAULT_LIMIT", "5")), ge=1, le=20)
    context_chars: int = Field(
        default=int(os.getenv("RAG_CONTEXT_CHARS", "2000")), ge=100, le=10000
    )
    include_sources: bool = Field(default=_getenv_bool("RAG_INCLUDE_SOURCES", True))
    with_chunk_text: bool = Field(default=_getenv_bool("RAG_WITH_CHUNK_TEXT", False))
    include_debug: bool = Field(default=_getenv_bool("RAG_INCLUDE_DEBUG", False))
    min_term_hits: int = Field(
        default=int(os.getenv("RAG_MIN_TERM_HITS", "0")),
        ge=0,
        description="Минимум вхождений редких токенов запроса",
    )
    diversify: bool = Field(
        default=_getenv_bool("RAG_DIVERSIFY", True),
        description="Включить MMR-диверсификацию по токенам",
    )
    mmr_lambda: float = Field(
        default=float(os.getenv("RAG_MMR_LAMBDA", "0.7")),
        ge=0.0,
        le=1.0,
        description="Баланс релевантности и разнообразия для MMR",
    )
    internal_candidates: Optional[int] = Field(
        default=(
            int(os.getenv("RAG_INTERNAL_CANDIDATES"))
            if os.getenv("RAG_INTERNAL_CANDIDATES")
            else None
        ),
        ge=1,
        le=500,
        description="Переопределить число внутренних кандидатов",
    )
    stitch_neighbors: bool = Field(
        default=_getenv_bool("RAG_STITCH_NEIGHBORS", True),
        description="Сшивать соседние чанки вокруг хита",
    )
    stitch_before: int = Field(
        default=int(os.getenv("RAG_STITCH_BEFORE", "2")),
        ge=0,
        le=20,
        description="Сколько чанков взять до хита",
    )
    stitch_after: int = Field(
        default=int(os.getenv("RAG_STITCH_AFTER", "8")),
        ge=0,
        le=20,
        description="Сколько чанков взять после хита",
    )
    stitch_use_chunk_text: bool = Field(
        default=_getenv_bool("RAG_STITCH_USE_CHUNK_TEXT", True),
        description="При сшивании использовать исходный chunk_text (иначе text)",
    )
    align_with_search: bool = Field(
        default=_getenv_bool(
            "RAG_ALIGN_WITH_SEARCH",
            False),
        description="Привести ранжирование ближе к /v1/search: отключить diversify, расширить окно сшивания, активнее включить точные и fuzzy фичи",
    )
    adaptive_expand: bool = Field(
        default=_getenv_bool(
            "RAG_ADAPTIVE_EXPAND",
            True),
        description="Авторасширять внутренний пул кандидатов, если нет сильных лексико-фразовых сигналов",
    )
    require_term_hit: bool = Field(
        default=_getenv_bool(
            "RAG_REQUIRE_TERM_HIT",
            True),
        description="Гарантировать, что хотя бы один выбранный фрагмент содержит сильное вхождение запроса (точное/приближенное/фаззи) в title/text",
    )


class AutoRequest(BaseModel):
    collection: Optional[str] = None
    points: Optional[List[PointIn]] = None
    delete_ids: Optional[List[int | str]] = None
    texts: Optional[List[str]] = None
    query: Optional[str] = None

    wait: bool = False

    limit: int = Field(default=5, ge=1, le=50)
    context_chars: int = Field(default=2000, ge=100, le=10000)
    with_chunk_text: bool = False
    include_sources: bool = True
    include_debug: bool = False
    min_term_hits: int = Field(default=0, ge=0)
    diversify: bool = True
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    internal_candidates: Optional[int] = Field(default=None, ge=1, le=500)
