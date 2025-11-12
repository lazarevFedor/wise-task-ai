from __future__ import annotations

from typing import Dict, Any, Set

from token_utils import _extract_tokens

__all__ = [
    "text_tokens",
    "_has_term_hit",
]

def text_tokens(entry: Dict[str, Any]) -> Set[str]:
    pl = entry.get("payload", {}) if isinstance(entry, dict) else {}
    return set(_extract_tokens(str(pl.get("text") or "")))

def _has_term_hit(e: Dict[str, Any]) -> bool:
    return bool(
        e.get("phrase_in_title") or e.get("phrase_in_text")
        or (e.get("lex_matches", 0) >= 1)
        or (e.get("approx_title_hits", 0) >= 1)
        or (e.get("fuzzy_title_hits", 0) >= 1)
        or (e.get("fuzzy_text_hits", 0) >= 1)
    )
