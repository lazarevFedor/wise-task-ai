from __future__ import annotations

import os
import re
from typing import Optional, List

from token_utils import _extract_tokens

__all__ = [
    "_expected_source_for_query",
    "_expected_source_for_title",
    "_derive_phrases",
]


def _expected_source_for_query(query: str) -> Optional[str]:
    m = re.search(r"алгоритм\s+([A-Za-zА-Яа-яЁё\-]+)", query or "", flags=re.IGNORECASE)
    if not m:
        return None
    name = m.group(1).strip()
    if not name:
        return None
    name_cap = name[:1].upper() + name[1:]
    return f"Просмотр_исходного_текста_страницы_Алгоритм_{name_cap}.tex"


def _expected_source_for_title(query: str) -> Optional[str]:
    q = (query or "").strip()
    if not q:
        return None
    if not re.search(r"[A-Za-zА-Яа-яЁё]", q):
        return None
    words = [w for w in re.split(r"[\s\-]+", q) if w]
    if not (1 <= len(words) <= 4):
        return None

    def _cap(w: str) -> str:
        return (w[:1].upper() + w[1:]) if w else w

    joined = "_".join(_cap(w) for w in words)
    return f"Просмотр_исходного_текста_страницы_{joined}.tex"


def _derive_phrases(query: str) -> List[str]:
    q = (query or "").strip().lower()
    if not q:
        return []
    toks = _extract_tokens(q)
    if not toks:
        return []
    try:
        max_ng = int(os.getenv("PHRASE_MAX_NGRAM", "3"))
    except Exception:
        max_ng = 3
    try:
        min_len = int(os.getenv("PHRASE_MIN_LEN", "6"))
    except Exception:
        min_len = 6
    extra_sw = {
        s.strip().lower()
        for s in (
            os.getenv("PHRASE_STOPWORDS", "").split(",")
            if os.getenv("PHRASE_STOPWORDS")
            else []
        )
        if s.strip()
    }
    phrases: List[str] = []
    seen: set[str] = set()
    for n in range(2, max_ng + 1):
        if len(toks) < n:
            break
        for i in range(0, len(toks) - n + 1):
            gram = toks[i: i + n]
            if any(t in extra_sw for t in gram):
                continue
            phrase = " ".join(gram)
            if phrase in seen:
                continue
            if len(phrase) < min_len:
                continue
            if len(gram) > 4:
                continue
            seen.add(phrase)
            phrases.append(phrase)
    if not phrases:
        for t in toks:
            if len(t) >= min_len and t not in extra_sw and t not in seen:
                seen.add(t)
                phrases.append(t)
            if len(phrases) >= 2:
                break

    def _score(p: str) -> tuple[int, int]:
        parts = p.split()
        uniq_long = sum(1 for w in set(parts) if len(w) >= 5)
        return (uniq_long, len(p))

    phrases.sort(key=_score, reverse=True)
    return phrases[:2]
