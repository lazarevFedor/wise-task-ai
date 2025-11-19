from __future__ import annotations

from typing import List

import re

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")
STOPWORDS = {
    "алгоритм",
    "алгоритма",
    "алгоритмы",
    "алгоритмов",
    "algorithm",
}


def _extract_tokens(text: str) -> List[str]:
    if not text:
        return []
    toks: List[str] = []
    seen: set[str] = set()
    for m in WORD_RE.findall(text.lower()):
        if len(m) < 4:
            continue
        if m in STOPWORDS:
            continue
        if m in seen:
            continue
        seen.add(m)
        toks.append(m)
    return toks


def _approx_token_variants(token: str) -> List[str]:
    t = (token or "").lower()
    if not t:
        return []
    variants = {t}
    for suf in (
        "ами",
        "ями",
        "ями",
        "ями",
        "его",
        "ого",
        "ему",
        "ому",
        "ыми",
        "ими",
        "ых",
        "их",
        "ой",
        "ей",
        "ый",
        "ий",
        "ая",
        "ое",
        "ое",
        "ую",
        "eu",
        "ам",
        "ям",
        "ах",
        "ях",
        "ов",
        "ев",
        "ом",
        "ем",
        "ым",
        "им",
        "а",
        "я",
        "ы",
        "и",
        "у",
        "ю",
        "о",
        "е",
        "ё",
        "й",
        "ь",
    ):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            variants.add(t[: len(t) - len(suf)])
    return list(variants)


def _approx_in(haystack: str, token: str) -> bool:
    if not haystack or not token:
        return False
    h = haystack.lower()
    for v in _approx_token_variants(token):
        if v and v in h:
            return True
    return False


def _levenshtein_at_most(a: str, b: str, max_dist: int = 1) -> bool:
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > max_dist:
        return False
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    prev = list(range(la + 1))
    for j in range(1, lb + 1):
        cj = [j] + [0] * la
        bj = b[j - 1]
        start = max(1, j - max_dist)
        end = min(la, j + max_dist)
        if start > end:
            return False
        for i in range(1, start):
            cj[i] = max_dist + 1
        for i in range(start, end + 1):
            cost = 0 if a[i - 1] == bj else 1
            cj[i] = min(
                prev[i] + 1,
                cj[i - 1] + 1,
                prev[i - 1] + cost,
            )
        for i in range(end + 1, la + 1):
            cj[i] = max_dist + 1
        prev = cj
        if min(prev) > max_dist:
            return False
    return prev[la] <= max_dist


def _text_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", text or "")


__all__ = [
    "_extract_tokens",
    "_approx_token_variants",
    "_approx_in",
    "_levenshtein_at_most",
    "_text_words",
]
