from __future__ import annotations

import os
import threading
from typing import Any, List

try:
    from fastembed import TextEmbedding  
except Exception:  
    TextEmbedding = None  
try:
    from sentence_transformers import SentenceTransformer  
except Exception: 
    SentenceTransformer = None  

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/e5-small-v2")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "384"))

_model_lock = threading.Lock()
_embedding_model: Any = None

__all__ = ["_get_embedding_model", "_embed_query_text", "EMBEDDING_MODEL"]


def _get_embedding_model() -> Any:
    global _embedding_model
    if _embedding_model is None:
        with _model_lock:
            if _embedding_model is None:
                if TextEmbedding is not None:
                    fe = TextEmbedding(EMBEDDING_MODEL)
                    sample_vec = next(fe.embed(["probe"]))
                    dim = len(sample_vec)
                    if VECTOR_SIZE and dim != VECTOR_SIZE:
                        raise RuntimeError(f"VECTOR_SIZE={VECTOR_SIZE} != model dim={dim}")
                    _embedding_model = fe
                elif SentenceTransformer is not None:
                    st = SentenceTransformer(EMBEDDING_MODEL)
                    dim = st.get_sentence_embedding_dimension()
                    if VECTOR_SIZE and dim != VECTOR_SIZE:
                        raise RuntimeError(f"VECTOR_SIZE={VECTOR_SIZE} != model dim={dim}")
                    _embedding_model = st
                else:
                    raise RuntimeError("No embedding backend (fastembed or sentence-transformers) available")
    return _embedding_model


def _embed_query_text(query: str) -> List[float]:
    """Embed a single query string into vector of floats."""
    model = _get_embedding_model()
    try:
        if hasattr(model, "embed"):
            emb = list(model.embed([query]))[0]
        else:
            import numpy as np  
            arr = model.encode([query], show_progress_bar=False, convert_to_numpy=True)
            emb = arr[0] if isinstance(arr, np.ndarray) else list(arr)[0]
        try:
            return emb.tolist()  
        except Exception:
            return [float(x) for x in emb]
    except Exception as e:  
        raise RuntimeError(f"Embedding failed: {e}")
