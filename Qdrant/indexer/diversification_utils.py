from __future__ import annotations

from typing import List, Dict, Any, Set

__all__ = ["mmr_diversify"]


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    i = len(a & b)
    u = len(a | b)
    return (i / u) if u else 0.0


def mmr_diversify(
    *,
    candidates: List[Dict[str, Any]],
    limit: int,
    lambda_: float,
    tokenize_func,
    forced_top: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    selected: List[Dict[str, Any]] = []
    tokens_list = [tokenize_func(c) for c in candidates]
    while candidates and len(selected) < limit:
        best = None
        best_mmr = -1e9
        for idx, c in enumerate(candidates):
            rel = float(c.get("score", 0.0))
            if not selected:
                mmr = rel
            else:
                sim_max = 0.0
                c_tokens = tokens_list[idx]
                for s in selected:
                    s_tokens = s.get("_tokens", set())
                    sim = _jaccard(c_tokens, s_tokens)
                    if sim > sim_max:
                        sim_max = sim
                mmr = lambda_ * rel - (1 - lambda_) * sim_max
            if mmr > best_mmr:
                best_mmr = mmr
                best = (idx, c)
        if best is None:
            break
        bi, bc = best
        bc = dict(bc)
        bc["_tokens"] = tokens_list[bi]
        selected.append(bc)
        del candidates[bi]
        del tokens_list[bi]

    if forced_top and forced_top not in selected:
        selected = [forced_top] + selected
        if len(selected) > limit:
            selected = selected[:limit]
    return selected
