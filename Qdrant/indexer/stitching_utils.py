from __future__ import annotations

from typing import Dict, Any, List
import requests

__all__ = ["stitch_neighbors_for_hit", "stitch_neighbors_for_hits"]


def _scroll_range(
    *,
    qdrant_base_url: str,
    collection: str,
    source: str,
    gte_idx: int,
    lte_idx: int,
    limit: int,
    timeout: float = 8.0,
) -> List[Dict[str, Any]]:
    flt = {
        "must": [
            {"key": "source", "match": {"value": source}},
            {"key": "chunk_index", "range": {"gte": gte_idx, "lte": lte_idx}},
        ]
    }
    scroll_url = f"{qdrant_base_url}/collections/{collection}/points/scroll"
    s_resp = requests.post(
        scroll_url,
        json={
            "filter": flt,
            "with_payload": True,
            "limit": limit,
            "offset": None,
        },
        timeout=timeout,
    )
    s_resp.raise_for_status()
    s_data = s_resp.json()
    return s_data.get("result", {}).get("points", [])


def stitch_neighbors_for_hit(
    *,
    qdrant_base_url: str,
    collection: str,
    hit: Dict[str, Any],
    before: int,
    after: int,
    use_chunk_text: bool,
    max_chars: int | None = None,
    include_debug: bool = False,
) -> Dict[str, Any]:
    pl = hit.get("payload", {}) or {}
    src = pl.get("source")
    base_idx = pl.get("chunk_index")
    if src is None or base_idx is None:
        return hit
    gte_idx = max(0, int(base_idx) - int(before))
    lte_idx = int(base_idx) + int(after)
    try:
        points = _scroll_range(
            qdrant_base_url=qdrant_base_url,
            collection=collection,
            source=src,
            gte_idx=gte_idx,
            lte_idx=lte_idx,
            limit=(before + after + 8),
        )

        def _idx(p):
            return (p.get("payload", {}) or {}).get("chunk_index", 0)

        points.sort(key=_idx)
        pieces: List[str] = []
        for p in points:
            ppl = p.get("payload", {}) or {}
            if use_chunk_text:
                seg = ppl.get("chunk_text") or ""
            else:
                seg = ppl.get("text") or ppl.get("chunk_text") or ""
            if seg:
                pieces.append(str(seg))
        stitched_text = "\n\n".join(pieces)
        if max_chars is not None and max_chars > 0:
            stitched_text = stitched_text[:max_chars]
        new_hit = dict(hit)
        new_pl = dict(pl)
        new_pl["text"] = stitched_text
        new_hit["payload"] = new_pl
        if include_debug:
            dbg = new_hit.setdefault("debug", {})
            if isinstance(dbg, dict):
                dbg["stitched_count"] = len(points)
                dbg["stitched_range"] = [gte_idx, lte_idx]
        return new_hit
    except Exception:
        return hit


def stitch_neighbors_for_hits(
    *,
    qdrant_base_url: str,
    collection: str,
    hits: List[Dict[str, Any]],
    before: int,
    after: int,
    use_chunk_text: bool,
    max_chars: int | None = None,
    include_debug: bool = False,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for h in hits:
        out.append(
            stitch_neighbors_for_hit(
                qdrant_base_url=qdrant_base_url,
                collection=collection,
                hit=h,
                before=before,
                after=after,
                use_chunk_text=use_chunk_text,
                max_chars=max_chars,
                include_debug=include_debug,
            )
        )
    return out
