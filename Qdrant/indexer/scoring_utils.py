from __future__ import annotations

import re
from typing import Dict, Any, List, Optional

from token_utils import _extract_tokens, _approx_in
from phrase_utils import _derive_phrases

__all__ = [
    "_is_definition_query",
    "prepare_query_context",
    "compute_search_candidate_score",
    "compute_rag_candidate_score",
    "compute_rag_candidate_score_expansion_legacy",
]

def _is_definition_query(text: str) -> bool:
    q = (text or "").lower()
    return ("что такое" in q) or ("определен" in q) or ("definition" in q) or ("определи" in q)


def prepare_query_context(query: str) -> Dict[str, Any]:
    q = query or ""
    q_tokens: List[str] = _extract_tokens(q)
    phrases: List[str] = _derive_phrases(q)
    definitional: bool = _is_definition_query(q)
    algo_query: bool = ("алгоритм" in q.lower())
    title_focus_token: Optional[str] = None
    try:
        m = re.search(r"алгоритм\s+([A-Za-zА-Яа-яЁё\-]+)", q, flags=re.IGNORECASE)
        if m:
            title_focus_token = m.group(1).strip()
    except Exception:
        title_focus_token = None
    phrase_lower = q.strip().lower()
    return {
        "q_tokens": q_tokens,
        "phrases": phrases,
        "definitional": definitional,
        "algo_query": algo_query,
        "title_focus_token": title_focus_token,
        "phrase_lower": phrase_lower,
    }


def compute_search_candidate_score(
    *,
    raw: Dict[str, Any],
    payload: Dict[str, Any],
    query: str,
    qctx: Dict[str, Any],
    with_chunk_text: bool,
    max_chars: Optional[int],
    include_debug: bool,
    expected_source: Optional[str],
) -> Dict[str, Any]:
    from config_utils import (
        W_VECTOR,
        W_LEX_MATCH,
        TH_LEX_MATCH_CAP,
        W_TITLE_MATCH,
        W_SOURCE_MATCH,
        W_APPROX_TITLE_HITS,
        TH_APPROX_TITLE_HITS_CAP,
        W_FUZZY_TITLE_HITS,
        TH_FUZZY_TITLE_CAP,
        W_FUZZY_TEXT_HITS,
        TH_FUZZY_TEXT_CAP,
        W_TITLE_PHRASE,
        W_TEXT_PHRASE,
        W_ALGO_NAME_MATCH,
        W_DEF_BOOST,
        W_EXACT_SOURCE_MATCH,
        W_DEF_LIKE,
        W_HAS_PROPERTIES,
        W_HAS_LEMMA_THEOREM,
        W_BARYCENTER_FOCUS,
        W_BOILERPLATE_PENALTY,
        W_EARLY_CHUNK_BONUS,
        TH_EARLY_CHUNK_INDEX,
    )
    from token_utils import _approx_in, _levenshtein_at_most, _text_words
    from phrase_utils import _expected_source_for_query

    vec_score = float(raw.get("score", 0.0) or 0.0)
    title = str(payload.get("title") or "")
    source = payload.get("source")
    chunk_idx = payload.get("chunk_index")
    text_raw = str(payload.get("chunk_text") or "") if with_chunk_text else str(payload.get("text") or payload.get("chunk_text") or "")
    text_lower = text_raw.lower()

    q_tokens = qctx["q_tokens"]
    phrases = qctx["phrases"]
    definitional = qctx["definitional"]
    algo_query = qctx["algo_query"]
    title_focus_token = qctx["title_focus_token"]

    matches = 0
    if q_tokens:
        for t in q_tokens:
            matches += text_lower.count(t)
    ttl = title.lower()
    title_match = 1 if (q_tokens and any(t in ttl for t in q_tokens)) else 0
    approx_title_hits = 0
    if q_tokens:
        for t in q_tokens:
            if _approx_in(ttl, t):
                approx_title_hits += 1
    phrase_in_title = 1 if (phrases and any(ph in ttl for ph in phrases)) else 0
    all_tokens_in_title = 1 if (q_tokens and all(t in ttl for t in q_tokens)) else 0
    phrase_in_text = 1 if (phrases and any(ph in text_lower for ph in phrases)) else 0

    fuzzy_title_hits = 0
    fuzzy_text_hits = 0
    if q_tokens:
        title_words = set(_text_words(ttl))
        tw = _text_words(text_lower)[:300]
        text_words = set(tw)
        for t in q_tokens:
            if len(t) >= 5 and any(_levenshtein_at_most(t, w, 1) for w in title_words):
                fuzzy_title_hits += 1
            if len(t) >= 5 and any(_levenshtein_at_most(t, w, 1) for w in text_words):
                fuzzy_text_hits += 1

    source_str = str(source or "").lower()
    source_match = 1 if (q_tokens and any(t in source_str for t in q_tokens)) else 0
    if expected_source is None:
        try:
            expected_source = _expected_source_for_query(query)
        except Exception:
            expected_source = None
    exact_source_match = 1 if (source and expected_source == source) else 0
    algo_name_match = 1 if (title_focus_token and _approx_in(ttl, title_focus_token)) else 0
    def_boost = 1 if (definitional and ("определен" in text_lower or "определен" in ttl or "definition" in text_lower or "definition" in ttl)) else 0
    def_like = 1 if (algo_query and (" — алгоритм" in text_lower or " это алгоритм" in text_lower)) else 0
    has_properties = 1 if ("основные свойства" in text_lower or "основные свойства" in ttl) else 0
    has_lemma_theorem = 1 if ("лемма" in text_lower or "теорема" in text_lower) else 0

    rare_focus_tokens = [t for t in q_tokens if len(t) >= 9]
    focus_present = any(t in ttl or t in source_str for t in rare_focus_tokens)
    barycenter_focus = 1 if focus_present else 0
    bary_bonus = (3.5 if (rare_focus_tokens and focus_present) else 0.0)
    bary_miss_penalty = (-2.2 if (rare_focus_tokens and not focus_present) else 0.0)
    boilerplate = 1 if any(k in text_lower for k in (
        "источники информации", "см. также", "категория:", "== реализация ==", "== корректность ==", "== идея ==", "== оценка производительности =="
    )) else 0

    final_score = (
        vec_score * W_VECTOR
        + W_LEX_MATCH * min(matches, TH_LEX_MATCH_CAP)
        + W_TITLE_MATCH * title_match
        + W_SOURCE_MATCH * source_match
        + W_APPROX_TITLE_HITS * min(approx_title_hits, TH_APPROX_TITLE_HITS_CAP)
        + W_FUZZY_TITLE_HITS * min(fuzzy_title_hits, TH_FUZZY_TITLE_CAP)
        + W_FUZZY_TEXT_HITS * min(fuzzy_text_hits, TH_FUZZY_TEXT_CAP)
        + W_TITLE_PHRASE * max(phrase_in_title, all_tokens_in_title)
        + W_TEXT_PHRASE * phrase_in_text
        + W_ALGO_NAME_MATCH * algo_name_match
        + W_DEF_BOOST * def_boost
        + W_EXACT_SOURCE_MATCH * exact_source_match
        + (W_DEF_LIKE if def_like else 0.0)
        + (W_HAS_PROPERTIES if has_properties else 0.0)
        + (W_HAS_LEMMA_THEOREM if has_lemma_theorem else 0.0)
        + (W_BARYCENTER_FOCUS if barycenter_focus else 0.0)
        + bary_bonus
        + bary_miss_penalty
        + (W_BOILERPLATE_PENALTY if (algo_query and boilerplate) else 0.0)
        + (W_EARLY_CHUNK_BONUS if (algo_query and (payload.get("chunk_index", 9999) is not None and int(payload.get("chunk_index", 9999)) <= TH_EARLY_CHUNK_INDEX)) else 0.0)
    )

    text_out = text_raw
    if max_chars is not None and max_chars > 0:
        text_out = text_out[: max_chars]

    result = {
        "id": raw.get("id"),
        "score": final_score,
        "payload": {
            "title": title,
            "source": source,
            "chunk_index": chunk_idx,
            "text": text_out,
        },
    }
    if include_debug:
        result["debug"] = {
            "vector_score": vec_score,
            "lex_matches": matches,
            "title_match": bool(title_match),
            "source_match": bool(source_match),
            "exact_source_match": bool(exact_source_match),
            "approx_title_hits": approx_title_hits,
            "fuzzy_title_hits": fuzzy_title_hits,
            "fuzzy_text_hits": fuzzy_text_hits,
            "algo_name_match": bool(algo_name_match),
            "def_boost": bool(def_boost),
            "def_like": bool(def_like),
            "has_properties": bool(has_properties),
            "has_lemma_theorem": bool(has_lemma_theorem),
            "barycenter_focus": bool(barycenter_focus),
            "boilerplate": bool(boilerplate),
            "final_score": final_score,
        }
    return result


def compute_rag_candidate_score(
    *,
    raw: Dict[str, Any],
    payload: Dict[str, Any],
    query: str,
    qctx: Dict[str, Any],
    with_chunk_text: bool,
    context_chars: int,
    include_debug: bool,
    expected_source: str | None,
) -> Dict[str, Any]:
    from config_utils import (
        W_VECTOR,
        W_LEX_MATCH,
        TH_LEX_MATCH_CAP,
        W_TITLE_MATCH,
        W_SOURCE_MATCH,
        W_APPROX_TITLE_HITS,
        TH_APPROX_TITLE_HITS_CAP,
        W_FUZZY_TITLE_HITS,
        TH_FUZZY_TITLE_CAP,
        W_FUZZY_TEXT_HITS,
        TH_FUZZY_TEXT_CAP,
        W_TITLE_PHRASE,
        W_TEXT_PHRASE,
        W_ALGO_NAME_MATCH,
        W_DEF_BOOST,
        W_EXACT_SOURCE_MATCH,
        W_DEF_LIKE,
        W_BOILERPLATE_PENALTY,
        W_EARLY_CHUNK_BONUS,
        TH_EARLY_CHUNK_INDEX,
        W_IS_DEFINITION_FLAG,
        W_HAS_MATH_FLAG,
        W_ALGO_NAME_PAYLOAD_MATCH,
        W_IS_ALGORITHM_FLAG,
    )
    from token_utils import _approx_in, _levenshtein_at_most, _text_words
    from phrase_utils import _expected_source_for_query

    vec_score = float(raw.get("score", 0.0) or 0.0)
    title = str(payload.get("title") or "")
    source = payload.get("source")
    chunk_idx = payload.get("chunk_index")
    text_raw = str(payload.get("chunk_text") or "") if with_chunk_text else str(payload.get("text") or payload.get("chunk_text") or "")
    low = text_raw.lower()

    q_tokens = qctx["q_tokens"]
    phrases = qctx["phrases"]
    definitional = qctx["definitional"]
    algo_query = qctx["algo_query"]
    phrase_lower = qctx["phrase_lower"]
    title_focus_token = qctx["title_focus_token"]

    matches = 0
    if q_tokens:
        for t in q_tokens:
            matches += low.count(t)
    ttl = title.lower()
    title_match = 1 if (q_tokens and any(t in ttl for t in q_tokens)) else 0
    phrase_in_title = 1 if (phrases and any(ph in ttl for ph in phrases)) else 0
    all_tokens_in_title = 1 if (q_tokens and all(t in ttl for t in q_tokens)) else 0
    phrase_in_text = 1 if (phrases and any(ph in low for ph in phrases)) else 0
    approx_title_hits = 0
    if q_tokens:
        for t in q_tokens:
            if _approx_in(ttl, t):
                approx_title_hits += 1
    fuzzy_title_hits = 0
    fuzzy_text_hits = 0
    if q_tokens:
        title_words = set(_text_words(ttl))
        tw = _text_words(low)[:300]
        text_words = set(tw)
        for t in q_tokens:
            if len(t) >= 5 and any(_levenshtein_at_most(t, w, 1) for w in title_words):
                fuzzy_title_hits += 1
            if len(t) >= 5 and any(_levenshtein_at_most(t, w, 1) for w in text_words):
                fuzzy_text_hits += 1
    source_str = str(source or "").lower()
    source_match = 1 if (q_tokens and any(t in source_str for t in q_tokens)) else 0
    algo_name_match = 1 if (title_focus_token and _approx_in(ttl, title_focus_token)) else 0
    def_boost = 1 if (definitional and ("определен" in low or "определен" in ttl or "definition" in low or "definition" in ttl)) else 0
    if expected_source is None:
        try:
            expected_source = _expected_source_for_query(query)
        except Exception:
            expected_source = None
    exact_source_match = 1 if (source and expected_source == source) else 0
    def_like = 1 if (algo_query and (" — алгоритм" in low or " это алгоритм" in low)) else 0
    boilerplate = 1 if any(k in low for k in (
        "источники информации", "см. также", "категория:", "== реализация ==", "== корректность ==", "== идея ==", "== оценка производительности =="
    )) else 0
    is_definition_flag = 1 if payload.get("is_definition") else 0
    is_algorithm_flag = 1 if payload.get("is_algorithm") else 0
    has_math_flag = 1 if payload.get("has_math") else 0
    algo_name_payload_match = 1 if (title_focus_token and payload.get("algorithm_name") and _approx_in(str(payload.get("algorithm_name")), title_focus_token)) else 0

    final_score = (
        vec_score * W_VECTOR
        + W_LEX_MATCH * min(matches, TH_LEX_MATCH_CAP)
        + W_TITLE_MATCH * title_match
        + W_SOURCE_MATCH * source_match
        + W_APPROX_TITLE_HITS * min(approx_title_hits, TH_APPROX_TITLE_HITS_CAP)
        + W_FUZZY_TITLE_HITS * min(fuzzy_title_hits, TH_FUZZY_TITLE_CAP)
        + W_FUZZY_TEXT_HITS * min(fuzzy_text_hits, TH_FUZZY_TEXT_CAP)
        + W_TITLE_PHRASE * max(phrase_in_title, all_tokens_in_title)
        + W_TEXT_PHRASE * phrase_in_text
        + W_ALGO_NAME_MATCH * algo_name_match
        + W_DEF_BOOST * def_boost
        + W_EXACT_SOURCE_MATCH * exact_source_match
        + (W_IS_DEFINITION_FLAG if (definitional and is_definition_flag) else 0.0)
        + (W_HAS_MATH_FLAG if (algo_query and has_math_flag) else 0.0)
        + (W_ALGO_NAME_PAYLOAD_MATCH if algo_name_payload_match else 0.0)
        + (W_IS_ALGORITHM_FLAG if (algo_query and is_algorithm_flag) else 0.0)
        + (W_DEF_LIKE if def_like else 0.0)
        + (W_BOILERPLATE_PENALTY if (algo_query and boilerplate) else 0.0)
        + (W_EARLY_CHUNK_BONUS if (algo_query and (payload.get("chunk_index", 9999) is not None and int(payload.get("chunk_index", 9999)) <= TH_EARLY_CHUNK_INDEX)) else 0.0)
    )

    text_out = text_raw[: context_chars] if context_chars and context_chars > 0 else text_raw
    cand = {
        "id": raw.get("id"),
        "score": final_score,
        "vector_score": vec_score,
        "lex_matches": matches,
        "title_match": bool(title_match),
        "source_match": bool(source_match),
        "approx_title_hits": approx_title_hits,
        "fuzzy_title_hits": fuzzy_title_hits,
        "fuzzy_text_hits": fuzzy_text_hits,
        "phrase_in_title": bool(phrase_in_title),
        "phrase_in_text": bool(phrase_in_text),
        "algo_name_match": bool(algo_name_match),
        "def_boost": bool(def_boost),
        "is_definition_flag": bool(is_definition_flag),
        "is_algorithm_flag": bool(is_algorithm_flag),
        "has_math_flag": bool(has_math_flag),
        "algo_name_payload_match": bool(algo_name_payload_match),
        "def_like": bool(def_like),
        "boilerplate": bool(boilerplate),
        "payload": {
            "title": title,
            "source": source,
            "chunk_index": chunk_idx,
            "text": text_out,
        },
    }
    return cand


def compute_rag_candidate_score_expansion_legacy(
    *,
    raw: Dict[str, Any],
    payload: Dict[str, Any],
    query: str,
    qctx: Dict[str, Any],
    with_chunk_text: bool,
    context_chars: int,
    include_debug: bool,
    expected_source: str | None,
    phrase_lower: str,
) -> Dict[str, Any]:
    from config_utils import (
        W_VECTOR,
        W_LEX_MATCH,
        TH_LEX_MATCH_CAP,
        W_TITLE_MATCH,
        W_SOURCE_MATCH,
        W_APPROX_TITLE_HITS,
        TH_APPROX_TITLE_HITS_CAP,
        W_FUZZY_TITLE_HITS,
        TH_FUZZY_TITLE_CAP,
        W_FUZZY_TEXT_HITS,
        TH_FUZZY_TEXT_CAP,
        W_TITLE_PHRASE,
        W_TEXT_PHRASE,
        W_ALGO_NAME_MATCH,
        W_DEF_BOOST,
        W_EXACT_SOURCE_MATCH,
        W_DEF_LIKE,
        W_BOILERPLATE_PENALTY,
        W_EARLY_CHUNK_BONUS,
        TH_EARLY_CHUNK_INDEX,
        W_IS_DEFINITION_FLAG,
        W_HAS_MATH_FLAG,
        W_ALGO_NAME_PAYLOAD_MATCH,
        W_IS_ALGORITHM_FLAG,
    )
    from token_utils import _approx_in, _levenshtein_at_most, _text_words
    from phrase_utils import _expected_source_for_query

    vec_score = float(raw.get("score", 0.0) or 0.0)
    title = str(payload.get("title") or "")
    source = payload.get("source")
    chunk_idx = payload.get("chunk_index")
    text_raw = str(payload.get("chunk_text") or "") if with_chunk_text else str(payload.get("text") or payload.get("chunk_text") or "")
    low = text_raw.lower()

    q_tokens = qctx["q_tokens"]
    definitional = qctx["definitional"]
    algo_query = qctx["algo_query"]
    title_focus_token = qctx["title_focus_token"]

    matches = 0
    if q_tokens:
        for t in q_tokens:
            matches += low.count(t)
    ttl = title.lower()
    title_match = 1 if (q_tokens and any(t in ttl for t in q_tokens)) else 0
    phrase_in_title = 1 if (phrase_lower and phrase_lower in ttl) else 0
    all_tokens_in_title = 1 if (q_tokens and all(t in ttl for t in q_tokens)) else 0
    phrase_in_text = 1 if (phrase_lower and phrase_lower in low) else 0
    approx_title_hits = 0
    if q_tokens:
        for t in q_tokens:
            if _approx_in(ttl, t):
                approx_title_hits += 1
    fuzzy_title_hits = 0
    fuzzy_text_hits = 0
    if q_tokens:
        title_words = set(_text_words(ttl))
        tw = _text_words(low)[:300]
        text_words = set(tw)
        for t in q_tokens:
            if len(t) >= 5 and any(_levenshtein_at_most(t, w, 1) for w in title_words):
                fuzzy_title_hits += 1
            if len(t) >= 5 and any(_levenshtein_at_most(t, w, 1) for w in text_words):
                fuzzy_text_hits += 1
    source_str = str(source or "").lower()
    source_match = 1 if (q_tokens and any(t in source_str for t in q_tokens)) else 0
    algo_name_match = 1 if (title_focus_token and _approx_in(ttl, title_focus_token)) else 0
    def_boost = 1 if (definitional and ("определен" in low or "определен" in ttl or "definition" in low or "definition" in ttl)) else 0
    if expected_source is None:
        try:
            expected_source = _expected_source_for_query(query)
        except Exception:
            expected_source = None
    exact_source_match = 1 if (source and expected_source == source) else 0
    def_like = 1 if (algo_query and (" — алгоритм" in low or " это алгоритм" in low)) else 0
    boilerplate = 1 if any(k in low for k in (
        "источники информации", "см. также", "категория:", "== реализация ==", "== корректность ==", "== идея ==", "== оценка производительности =="
    )) else 0
    is_definition_flag = 1 if payload.get("is_definition") else 0
    is_algorithm_flag = 1 if payload.get("is_algorithm") else 0
    has_math_flag = 1 if payload.get("has_math") else 0
    algo_name_payload_match = 1 if (title_focus_token and payload.get("algorithm_name") and _approx_in(str(payload.get("algorithm_name")), title_focus_token)) else 0

    final_score = (
        vec_score * W_VECTOR
        + W_LEX_MATCH * min(matches, TH_LEX_MATCH_CAP)
        + W_TITLE_MATCH * title_match
        + W_SOURCE_MATCH * source_match
        + W_APPROX_TITLE_HITS * min(approx_title_hits, TH_APPROX_TITLE_HITS_CAP)
        + W_FUZZY_TITLE_HITS * min(fuzzy_title_hits, TH_FUZZY_TITLE_CAP)
        + W_FUZZY_TEXT_HITS * min(fuzzy_text_hits, TH_FUZZY_TEXT_CAP)
        + W_TITLE_PHRASE * max(phrase_in_title, all_tokens_in_title)
        + W_TEXT_PHRASE * phrase_in_text
        + W_ALGO_NAME_MATCH * algo_name_match
        + W_DEF_BOOST * def_boost
        + W_EXACT_SOURCE_MATCH * exact_source_match
        + (W_IS_DEFINITION_FLAG if (definitional and is_definition_flag) else 0.0)
        + (W_HAS_MATH_FLAG if (algo_query and has_math_flag) else 0.0)
        + (W_ALGO_NAME_PAYLOAD_MATCH if algo_name_payload_match else 0.0)
        + (W_IS_ALGORITHM_FLAG if (algo_query and is_algorithm_flag) else 0.0)
        + (W_DEF_LIKE if def_like else 0.0)
        + (W_BOILERPLATE_PENALTY if (algo_query and boilerplate) else 0.0)
        + (W_EARLY_CHUNK_BONUS if (algo_query and (payload.get("chunk_index", 9999) is not None and int(payload.get("chunk_index", 9999)) <= TH_EARLY_CHUNK_INDEX)) else 0.0)
    )

    text_out = text_raw[: context_chars] if context_chars and context_chars > 0 else text_raw
    cand = {
        "id": raw.get("id"),
        "score": final_score,
        "vector_score": vec_score,
        "lex_matches": matches,
        "title_match": bool(title_match),
        "source_match": bool(source_match),
        "approx_title_hits": approx_title_hits,
        "fuzzy_title_hits": fuzzy_title_hits,
        "fuzzy_text_hits": fuzzy_text_hits,
        "phrase_in_title": bool(phrase_in_title),
        "phrase_in_text": bool(phrase_in_text),
        "algo_name_match": bool(algo_name_match),
        "def_boost": bool(def_boost),
        "is_definition_flag": bool(is_definition_flag),
        "is_algorithm_flag": bool(is_algorithm_flag),
        "has_math_flag": bool(has_math_flag),
        "algo_name_payload_match": bool(algo_name_payload_match),
        "def_like": bool(def_like),
        "boilerplate": bool(boilerplate),
        "payload": {
            "title": title,
            "source": source,
            "chunk_index": chunk_idx,
            "text": text_out,
        },
    }
    return cand
