from __future__ import annotations

from typing import Dict, Any, List, Optional
import requests

from phrase_utils import _expected_source_for_query, _expected_source_for_title, _derive_phrases

__all__ = [
    "inject_canonical_and_title_sources",
]


def inject_canonical_and_title_sources(
    *,
    query: str,
    base_results: List[Dict[str, Any]],
    qdrant_url: str,
    vector: List[float] | Any,
    primary_limit: int,
    timeout: float = 6.0,
    extra_limit: int = 40,
) -> List[Dict[str, Any]]:
    results = list(base_results)
    base_ids = {r.get("id") for r in results}
    try:
        expected_source = _expected_source_for_query(query)
        if expected_source:
            flt = {"must": [{"key": "source", "match": {"value": expected_source}}]}
            resp_focus = requests.post(
                qdrant_url,
                json={
                    "vector": vector,
                    "limit": extra_limit,
                    "with_payload": True,
                    "filter": flt,
                },
                timeout=timeout,
            )
            if resp_focus.ok:
                js = resp_focus.json()
                for r in js.get("result", [])[:extra_limit]:
                    if r.get("id") not in base_ids:
                        results.append(r)
    except Exception:
        pass
    # Derived title/phrase guesses
    try:
        src_candidates: List[str] = []
        et = _expected_source_for_title(query)
        if et:
            src_candidates.append(et)
        for ph in _derive_phrases(query)[:2]:
            etp = _expected_source_for_title(ph)
            if etp and etp not in src_candidates:
                src_candidates.append(etp)
        for guess in src_candidates:
            try:
                flt2 = {"must": [{"key": "source", "match": {"value": guess}}]}
                resp2 = requests.post(
                    qdrant_url,
                    json={
                        "vector": vector,
                        "limit": extra_limit,
                        "with_payload": True,
                        "filter": flt2,
                    },
                    timeout=timeout,
                )
                if resp2.ok:
                    js2 = resp2.json()
                    for r in js2.get("result", [])[:extra_limit]:
                        if r.get("id") not in base_ids:
                            results.append(r)
            except Exception:
                pass
    except Exception:
        pass
    return results
