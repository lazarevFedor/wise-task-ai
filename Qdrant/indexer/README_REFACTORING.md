- `token_utils.py`: tokenization, approximate matching, Levenshtein banded distance.
- `phrase_utils.py`: phrase derivation and source filename guessing helpers.
- `context_utils.py`: `_sanitize_context` wiki/markup cleanup.
- `config_utils.py`: environment-driven defaults and weight/threshold constants.
- `embedding_utils.py`: lazy embedding backend initialization and query embedding.
- `collection_utils.py`: Qdrant collection distance resolution and ID normalization (+ ensure_collection).
- `rag_utils.py`: small RAG helpers (`text_tokens`, `_has_term_hit`).
- `scoring_utils.py`: common query context (`prepare_query_context`),
- search scoring (`compute_search_candidate_score`) and RAG scoring (`compute_rag_candidate_score`).
- `source_injection.py`: canonical/title-based source focus injection for Qdrant results.
- `expansion_utils.py`: adaptive expansion of candidate pool when lexical/fuzzy signals are weak.
- `diversification_utils.py`: MMR-based diversification over token sets.
- `stitching_utils.py`: neighbor stitching around a hit (scroll + concatenate text).

- `schemas.py`: Pydantic request models reused across endpoints (`PointIn`, `UpsertRequest`, `DeleteRequest`, `EmbedRequest`, `SearchRequest`, `RagRequest`, `AutoRequest`).
- `search_rag.py`: pipeline functions that encapsulate endpoint orchestration without behavior changes (`run_search_pipeline`, `run_rag_pipeline`).

`ingest_service.py` now focuses on FastAPI endpoints and delegates request validation to `schemas.py` and orchestration to the pipeline module (`search_rag.py`), while all feature logic lives in dedicated utils. Public API behavior is preserved.

