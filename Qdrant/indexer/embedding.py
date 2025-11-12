import os
from typing import Iterable, List

class EmbeddingModel:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.environ.get('EMBEDDING_MODEL', 'intfloat/e5-small-v2')
        self._backend = None
        self._impl = None
        self._dim = None
        self._init_backend()

    def _init_backend(self):
        tried: list[str] = []
        try:
            from fastembed import TextEmbedding  # type: ignore
            try:
                supported = set(TextEmbedding.list_supported_models())
                if self.model_name not in supported:
                    fallback_order = [
                        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        "paraphrase-multilingual-MiniLM-L12-v2",
                        "intfloat/multilingual-e5-large",
                        "intfloat/multilingual-e5-base",
                        "intfloat/multilingual-e5-small"
                    ]
                    chosen = None
                    for candidate in fallback_order:
                        if candidate in supported:
                            chosen = candidate
                            break
                    if chosen is None:
                        raise RuntimeError(f"Model '{self.model_name}' not supported by fastembed. Supported count={len(supported)}")
                    self.model_name = chosen
            except Exception:
                pass
            self._impl = TextEmbedding(self.model_name)
            vec = next(self._impl.embed(["probe"]))
            self._dim = len(vec)
            self._backend = 'fastembed'
            return
        except Exception as e:  
            tried.append(f"fastembed: {e}")
        try:
            from sentence_transformers import SentenceTransformer  
            model = SentenceTransformer(self.model_name)
            vecs = model.encode(["probe"], show_progress_bar=False)
            v0 = vecs[0] if isinstance(vecs, list) else vecs[0]
            self._dim = len(v0)
            self._impl = model
            self._backend = 'sentence-transformers'
            return
        except Exception as e:  
            tried.append(f"sentence-transformers: {e}")
        raise RuntimeError("No embedding backend available. Attempts: " + " | ".join(tried))

    @property
    def dimension(self) -> int:
        if self._dim is None:
            raise RuntimeError("Embedding dimension unknown")
        return self._dim

    def encode(self, texts: Iterable[str]) -> List[List[float]]:
        items = list(texts)
        if self._backend == 'fastembed':
            return [list(map(float, v)) for v in self._impl.embed(items)]
        if self._backend == 'sentence-transformers':
            import numpy as np
            arr = self._impl.encode(items, show_progress_bar=False, convert_to_numpy=True)
            if isinstance(arr, np.ndarray):
                return [v.tolist() for v in arr]
            return [list(map(float, v)) for v in arr]
        raise RuntimeError("Unsupported backend")
