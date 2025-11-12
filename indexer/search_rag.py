from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests

from collection_utils import _resolve_distance
from config_utils import (
	DEFAULT_SEARCH_LIMIT,
	DEFAULT_SEARCH_MAX_CHARS,
	DEFAULT_SEARCH_WITH_CHUNK_TEXT,
	DEFAULT_SEARCH_INCLUDE_DEBUG,
	DEFAULT_SEARCH_STITCH_NEIGHBORS,
	DEFAULT_SEARCH_STITCH_BEFORE,
	DEFAULT_SEARCH_STITCH_AFTER,
	DEFAULT_SEARCH_STITCH_USE_CHUNK_TEXT,
	SEARCH_INTERNAL_MAX,
	SEARCH_INTERNAL_MIN,
	SEARCH_INTERNAL_MULT,
	DEFAULT_COLLECTION,
)
from context_utils import _sanitize_context
from diversification_utils import mmr_diversify
from embedding_utils import _embed_query_text, EMBEDDING_MODEL
from expansion_utils import adaptive_expand_pool
from phrase_utils import _derive_phrases, _expected_source_for_query, _expected_source_for_title
from rag_utils import _has_term_hit, text_tokens as rag_text_tokens
from scoring_utils import (
	prepare_query_context,
	compute_search_candidate_score,
	compute_rag_candidate_score,
)
from source_injection import inject_canonical_and_title_sources
from stitching_utils import stitch_neighbors_for_hits
from token_utils import _extract_tokens
from schemas import SearchRequest, RagRequest



QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))


def run_search_pipeline(body: SearchRequest) -> Dict[str, Any]:
	collection = body.collection or DEFAULT_COLLECTION

	vector = _embed_query_text(body.query)
	qctx = prepare_query_context(body.query)
	qdrant_url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{collection}/points/search"
	internal_limit = min(SEARCH_INTERNAL_MAX, max(body.limit * SEARCH_INTERNAL_MULT, SEARCH_INTERNAL_MIN))
	search_payload = {
		"vector": vector,
		"limit": internal_limit,
		"with_payload": True,
	}

	try:
		resp = requests.post(qdrant_url, json=search_payload, timeout=10)
		resp.raise_for_status()
		data = resp.json()
		# Canonical page focus injection (preserve behavior)
		try:
			expected_source = _expected_source_for_query(body.query)
			if expected_source:
				flt = {"must": [{"key": "source", "match": {"value": expected_source}}]}
				resp_focus = requests.post(qdrant_url, json={
					"vector": vector,
					"limit": 30,
					"with_payload": True,
					"filter": flt,
				}, timeout=6)
				if resp_focus.ok:
					focus_js = resp_focus.json()
					base_ids = {r.get("id") for r in data.get("result", [])}
					for r in focus_js.get("result", [])[:30]:
						if r.get("id") not in base_ids:
							data.setdefault("result", []).append(r)
			# general title-based guess and derived phrases
			phrases = _derive_phrases(body.query)
			src_candidates: List[str] = []
			et = _expected_source_for_title(body.query)
			if et:
				src_candidates.append(et)
			for ph in phrases[:2]:
				etp = _expected_source_for_title(ph)
				if etp and etp not in src_candidates:
					src_candidates.append(etp)
			for src_guess in src_candidates:
				try:
					flt2 = {"must": [{"key": "source", "match": {"value": src_guess}}]}
					resp_focus2 = requests.post(qdrant_url, json={
						"vector": vector,
						"limit": 30,
						"with_payload": True,
						"filter": flt2,
					}, timeout=6)
					if resp_focus2.ok:
						focus_js2 = resp_focus2.json()
						base_ids = {r.get("id") for r in data.get("result", [])}
						for r in focus_js2.get("result", [])[:30]:
							if r.get("id") not in base_ids:
								data.setdefault("result", []).append(r)
				except Exception:
					pass
		except Exception:
			pass

		# Rerank via compute_search_candidate_score
		enriched = []
		expected_source_cache = _expected_source_for_query(body.query)
		for raw in data.get("result", []):
			payload = raw.get("payload", {}) or {}
			out_hit = compute_search_candidate_score(
				raw=raw,
				payload=payload,
				query=body.query,
				qctx=qctx,
				with_chunk_text=body.with_chunk_text,
				max_chars=body.max_chars,
				include_debug=body.include_debug,
				expected_source=expected_source_cache,
			)
			enriched.append(out_hit)

		# Rerank and cut to user limit
		enriched.sort(key=lambda h: h.get("score", 0.0), reverse=True)
		hits = enriched[: body.limit]

		# Optionally stitch neighbors for each hit
		if body.stitch_neighbors and hits:
			import requests as _rq
			stitched_hits: List[Dict[str, Any]] = []
			for hit in hits:
				pl = hit.get("payload", {}) or {}
				src = pl.get("source")
				base_idx = pl.get("chunk_index")
				if src is None or base_idx is None:
					stitched_hits.append(hit)
					continue
				try:
					rng = {
						"gte": max(0, int(base_idx) - int(body.stitch_before)),
						"lte": int(base_idx) + int(body.stitch_after),
					}
					flt = {
						"must": [
							{"key": "source", "match": {"value": src}},
							{"key": "chunk_index", "range": rng},
						]
					}
					scroll_url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{collection}/points/scroll"
					s_resp = _rq.post(scroll_url, json={
						"filter": flt,
						"with_payload": True,
						"limit": (body.stitch_before + body.stitch_after + 8),
						"offset": None,
					}, timeout=8)
					s_resp.raise_for_status()
					s_data = s_resp.json()
					points = s_data.get("result", {}).get("points", [])
					def _idx(p):
						return (p.get("payload", {}) or {}).get("chunk_index", 0)
					points.sort(key=_idx)
					pieces: List[str] = []
					for p in points:
						ppl = p.get("payload", {}) or {}
						if body.stitch_use_chunk_text:
							seg = ppl.get("chunk_text") or ""
						else:
							seg = ppl.get("text") or ppl.get("chunk_text") or ""
						if seg:
							pieces.append(str(seg))
					stitched_text = "\n\n".join(pieces)
					# Trim to max_chars if set
					if body.max_chars is not None and body.max_chars > 0:
						stitched_text = stitched_text[: body.max_chars]
					new_hit = dict(hit)
					new_pl = dict(pl)
					new_pl["text"] = stitched_text
					new_hit["payload"] = new_pl
					if body.include_debug:
						dbg = new_hit.setdefault("debug", {})
						if isinstance(dbg, dict):
							dbg["stitched_count"] = len(points)
							dbg["stitched_range"] = [rng["gte"], rng["lte"]]
					stitched_hits.append(new_hit)
				except Exception:
					stitched_hits.append(hit)
			hits = stitched_hits

		return {
			"collection": collection,
			"query": body.query,
			"results": hits,
			"count": len(hits),
		}
	except Exception as e:
		raise


def run_rag_pipeline(body: RagRequest) -> Dict[str, Any]:
	collection = body.collection or DEFAULT_COLLECTION

	# Step 1: compute embedding and get a large pool from Qdrant
	vector = _embed_query_text(body.query)
	qdrant_url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{collection}/points/search"
	RAG_INTERNAL_MULT = int(os.getenv("RAG_INTERNAL_MULT", "24"))
	RAG_INTERNAL_MIN = int(os.getenv("RAG_INTERNAL_MIN", "60"))
	RAG_INTERNAL_MAX = int(os.getenv("RAG_INTERNAL_MAX", "240"))
	internal_limit = body.internal_candidates or min(RAG_INTERNAL_MAX, max(body.limit * RAG_INTERNAL_MULT, RAG_INTERNAL_MIN))
	if body.align_with_search:
		internal_limit = max(internal_limit, min(SEARCH_INTERNAL_MAX, max(body.limit * SEARCH_INTERNAL_MULT, SEARCH_INTERNAL_MIN)))
	try:
		resp = requests.post(qdrant_url, json={
			"vector": vector,
			"limit": internal_limit,
			"with_payload": True,
		}, timeout=10)
		resp.raise_for_status()
		raw_data = resp.json()
		base_list = raw_data.get("result", []) or []
		base_list = inject_canonical_and_title_sources(
			query=body.query,
			base_results=base_list,
			qdrant_url=qdrant_url,
			vector=vector,
			primary_limit=internal_limit,
			extra_limit=40,
		)
		raw_data["result"] = base_list
		pool_initial = len(base_list)
	except Exception:
		raise

	# Step 2: hybrid relevance features
	qctx = prepare_query_context(body.query)
	q_tokens = qctx["q_tokens"]
	definitional = qctx["definitional"]
	algo_query = qctx["algo_query"]
	phrase = qctx["phrase_lower"]  
	title_focus_token = qctx["title_focus_token"]
	candidates: List[Dict[str, Any]] = []
	expected_source_cache = _expected_source_for_query(body.query)
	for raw in raw_data.get("result", []):
		payload = raw.get("payload", {}) or {}
		cand = compute_rag_candidate_score(
			raw=raw,
			payload=payload,
			query=body.query,
			qctx=qctx,
			with_chunk_text=body.with_chunk_text,
			context_chars=body.context_chars,
			include_debug=body.include_debug,
			expected_source=expected_source_cache,
		)
		candidates.append(cand)

	# Optional adaptive pool expansion
	expanded_pool = False
	pool_after_expand = len(raw_data.get("result", []) or [])
	try:
		has_strong = any(
			(c.get("phrase_in_title") if "phrase_in_title" in c else False)
			or (c.get("phrase_in_text") if "phrase_in_text" in c else False)
			or (c.get("lex_matches", 0) >= 1)
			or (c.get("approx_title_hits", 0) >= 1)
			or (c.get("fuzzy_title_hits", 0) >= 1)
			or (c.get("fuzzy_text_hits", 0) >= 1)
			for c in candidates
		)
	except Exception:
		has_strong = False

	if body.adaptive_expand and not has_strong:
		# For legacy behavioral parity: during expansion phase use single full-phrase matching
		from scoring_utils import compute_rag_candidate_score_expansion_legacy

		def scoring_func(raw, payload, legacy: bool):
			if legacy:
				return compute_rag_candidate_score_expansion_legacy(
					raw=raw,
					payload=payload,
					query=body.query,
					qctx=qctx,
					with_chunk_text=body.with_chunk_text,
					context_chars=body.context_chars,
					include_debug=body.include_debug,
					expected_source=expected_source_cache,
					phrase_lower=phrase,
				)
			else:
				return compute_rag_candidate_score(
					raw=raw,
					payload=payload,
					query=body.query,
					qctx=qctx,
					with_chunk_text=body.with_chunk_text,
					context_chars=body.context_chars,
					include_debug=body.include_debug,
					expected_source=expected_source_cache,
				)

		candidates, expanded_pool, pool_after_expand = adaptive_expand_pool(
			active=True,
			candidates=candidates,
			raw_results=raw_data.get("result", []) or [],
			qdrant_url=qdrant_url,
			vector=vector,
			internal_limit=internal_limit,
			internal_max=RAG_INTERNAL_MAX,
			context_chars=body.context_chars,
			with_chunk_text=body.with_chunk_text,
			query_tokens=q_tokens,
			definitional=definitional,
			algo_query=algo_query,
			phrase_lower=phrase,
			title_focus_token=title_focus_token,
			expected_source_func=_expected_source_for_query,
			scoring_func=scoring_func,
		)

	# Step 3: optional filter by min_term_hits
	if body.min_term_hits > 0:
		candidates = [c for c in candidates if c.get("lex_matches", 0) >= body.min_term_hits or c.get("title_match")]

	# Sort by relevance first
	candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)

	# Step 4: MMR-like diversification on token sets
	diversify_flag = body.diversify and not body.align_with_search
	forced_top_hit = candidates[0] if (body.align_with_search and candidates) else None
	if diversify_flag:
		selected = mmr_diversify(
			candidates=list(candidates),  # pass a copy, function mutates
			limit=body.limit,
			lambda_=body.mmr_lambda,
			tokenize_func=rag_text_tokens,
			forced_top=forced_top_hit,
		)
	else:
		selected = candidates[: body.limit]
		if forced_top_hit and forced_top_hit not in selected:
			selected = [forced_top_hit] + selected
			selected = selected[: body.limit]

	# Ensure at least one explicit term hit is present in selected
	forced_term_hit = False
	if body.require_term_hit and selected and not any(_has_term_hit(s) for s in selected):
		for c in candidates:
			if c not in selected and _has_term_hit(c):
				selected = selected[:-1] + [c]
				forced_term_hit = True
				break

	# Step 5: optionally stitch neighbors for each selected hit
	stitch_after = body.stitch_after if not body.align_with_search else max(body.stitch_after, DEFAULT_SEARCH_STITCH_AFTER)
	stitch_before = body.stitch_before if not body.align_with_search else max(body.stitch_before, DEFAULT_SEARCH_STITCH_BEFORE)
	if body.stitch_neighbors and selected:
		qdrant_base_url = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
		selected = stitch_neighbors_for_hits(
			qdrant_base_url=qdrant_base_url,
			collection=collection,
			hits=selected,
			before=stitch_before,
			after=stitch_after,
			use_chunk_text=body.stitch_use_chunk_text,
			max_chars=body.context_chars,
			include_debug=body.include_debug,
		)

	# Step 6: assemble context and citations
	parts: List[str] = []
	citations: List[Dict[str, Any]] = []
	used = 0
	for i, hit in enumerate(selected, 1):
		pl = hit.get("payload", {})
		title = pl.get("title") or ""
		src = pl.get("source") or ""
		txt = pl.get("text") or ""
		chunk = pl.get("chunk_index")
		header = f"[{i}] {title} ({src}, chunk {chunk})".strip()
		snippet_raw = txt[: max(0, body.context_chars - used)] if body.context_chars else txt
		snippet = _sanitize_context(snippet_raw)
		parts.append(header + "\n" + snippet)
		citations.append({
			"rank": i,
			"id": hit.get("id"),
			"score": hit.get("score"),
			"title": title,
			"source": src,
			"chunk_index": chunk,
			"text": snippet,
			**({"debug": {
				"vector_score": hit.get("vector_score"),
				"lex_matches": hit.get("lex_matches"),
				"title_match": hit.get("title_match"),
				"source_match": hit.get("source_match"),
				"approx_title_hits": hit.get("approx_title_hits"),
				"fuzzy_title_hits": hit.get("fuzzy_title_hits"),
				"fuzzy_text_hits": hit.get("fuzzy_text_hits"),
				"phrase_in_title": hit.get("phrase_in_title"),
				"phrase_in_text": hit.get("phrase_in_text"),
				"algo_name_match": hit.get("algo_name_match"),
				"def_boost": hit.get("def_boost"),
				"def_like": hit.get("def_like"),
				"is_definition_flag": hit.get("is_definition_flag"),
				"is_algorithm_flag": hit.get("is_algorithm_flag"),
				"has_math_flag": hit.get("has_math_flag"),
				"algo_name_payload_match": hit.get("algo_name_payload_match"),
				"boilerplate": hit.get("boilerplate"),
			}} if body.include_debug else {})
		})
		used += len(snippet)
		if body.context_chars and used >= body.context_chars:
			break

	context = "\n\n".join(parts)
	response: Dict[str, Any] = {
		"query": body.query,
		"collection": collection,
		"context": context,
		"chunks": citations,
		"count": len(citations),
	}
	if body.include_sources:
		response["sources"] = [c.get("source") for c in citations]
	if body.include_debug:
		response["debug"] = {
			"pool_initial": locals().get("pool_initial"),
			"pool_after_expand": locals().get("pool_after_expand"),
			"expanded_pool": locals().get("expanded_pool", False),
			"internal_limit_used": internal_limit,
			"align_with_search": body.align_with_search,
			"adaptive_expand": body.adaptive_expand,
			"require_term_hit": body.require_term_hit,
			"forced_term_hit": locals().get("forced_term_hit", False),
			"selected_count": len(citations),
		}
	return response
