from __future__ import annotations

from typing import List, Dict, Any, Tuple
import requests

__all__ = ["adaptive_expand_pool"]


def _has_strong_signals(candidates: List[Dict[str, Any]]) -> bool:
    try:
        return any(
            (c.get("phrase_in_title") if "phrase_in_title" in c else False)
            or (c.get("phrase_in_text") if "phrase_in_text" in c else False)
            or (c.get("lex_matches", 0) >= 1)
            or (c.get("approx_title_hits", 0) >= 1)
            or (c.get("fuzzy_title_hits", 0) >= 1)
            or (c.get("fuzzy_text_hits", 0) >= 1)
            for c in candidates
        )
    except Exception:
        return False


def adaptive_expand_pool(
    *,
    active: bool,
    candidates: List[Dict[str, Any]],
    raw_results: List[Dict[str, Any]],
    qdrant_url: str,
    vector: List[float] | Any,
    internal_limit: int,
    internal_max: int,
    context_chars: int,
    with_chunk_text: bool,
    query_tokens: List[str],
    definitional: bool,
    algo_query: bool,
    phrase_lower: str,
    title_focus_token: str | None,
    expected_source_func,
    scoring_func,
) -> Tuple[List[Dict[str, Any]], bool, int]:
    pool_initial = len(raw_results)
    if not active or _has_strong_signals(candidates):
        return candidates, False, pool_initial
    try:
        extra_limit = min(500, max(internal_limit * 2, internal_limit + 80, internal_max))
        if extra_limit <= internal_limit:
            return candidates, False, pool_initial
        resp2 = requests.post(qdrant_url, json={
            "vector": vector,
            "limit": extra_limit,
            "with_payload": True,
        }, timeout=10)
        if not resp2.ok:
            return candidates, False, pool_initial
        js2 = resp2.json()
        base_ids = {r.get("id") for r in raw_results}
        new_results = [r for r in js2.get("result", []) or [] if r.get("id") not in base_ids]
        for raw in new_results:
            payload = raw.get("payload", {}) or {}
            cand = scoring_func(raw, payload, legacy=True)
            candidates.append(cand)
        pool_after_expand = pool_initial + len(js2.get("result", []) or [])
        return candidates, True, pool_after_expand
    except Exception:
        return candidates, False, pool_initial
