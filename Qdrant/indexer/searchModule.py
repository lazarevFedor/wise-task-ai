import os
import re
import pickle
from typing import List, Dict, Set
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from qdrant_client import models
import logging
logger = logging.getLogger(__name__)

class Searcher:
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "latex_books",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        bm25_index_path: str = "/app/data/bm25_index.pkl",
        alpha: float = 0.6,
    ):
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection_name = collection_name
        self.bm25_index_path = bm25_index_path
        self.alpha = alpha

        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.embedding_model = TextEmbedding(embedding_model)
        self.bm25 = None
        self.corpus = []
        self._load_bm25()

    def _load_bm25(self):
        if os.path.exists(self.bm25_index_path):
            logger.info(f"Загружаем BM25 индекс из {self.bm25_index_path}...")
            with open(self.bm25_index_path, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.corpus = data["corpus"]
            logger.info(f"BM25 индекс загружен ({len(self.corpus)} документов)")
        else:
            logger.info(f"BM25 индекс не найден. Поиск будет работать только через Qdrant.")

    def _tokenize_russian(self, text: str):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return tokens

    def get_collection_info(self) -> Dict:
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": getattr(info, 'vectors_count', None),
                "points_count": getattr(info, 'points_count', None),
                "status": getattr(info, 'status', None),
            }
        except Exception as e:
            return {"error": str(e)}
    def count_word_hits(self, text, query_words):
        words = set(re.findall(r'\b\w+\b', text.lower()))
        return len(words & query_words)
    def get_titles(self) -> List[str]:
        res = self.client.scroll(collection_name=self.collection_name, limit=10000)
        return list({point.payload.get("title", "") for point in res})

    def highlight_match(self, text: str, query: str) -> str:
        for word in query.split():
            text = re.sub(f'({re.escape(word)})', r'**\1**', text, flags=re.IGNORECASE)
        return text

    def search(
        self, query: str, limit: int = 10, score_threshold: float = 0.0
    ) -> List[Dict]:
        query_vector = list(self.embedding_model.embed([query]))[0]
        search_limit = limit * 10

        vector_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=search_limit,
            score_threshold=score_threshold,
            search_params=models.SearchParams(exact=True),
        )

        bm25_map = {}
        if self.bm25:
            tokenized_query = self._tokenize_russian(query)
            scores = self.bm25.get_scores(tokenized_query)
            if max(scores) > 0:
                scores = [s / max(scores) for s in scores]
            for doc_id, score in enumerate(scores):
                bm25_map[str(doc_id)] = score
        enriched_results = []
        query_words = set(re.findall(r'\b\w+\b', query.lower()))

        for idx, result in enumerate(vector_results):
            payload = result.payload
            vector_score = result.score
            text = payload.get("text", "")
            section = payload.get("section", "")
            title = payload.get("title", "").lower()
            bm25_score = bm25_map.get(str(result.id), 0.0)
            hybrid_score = self.alpha * vector_score + (1 - self.alpha) * bm25_score
            keyword_score = self._compute_keyword_score(query, payload)
            final_score = 0.5 * hybrid_score + 0.5 * keyword_score

            title_words = set(re.findall(r'\b\w+\b', title))
            title_matches = len(query_words & title_words)
            if title_matches > 0:
                if query.lower().replace(" ", "_") in title:
                    final_score *= 2.0
                else:
                    final_score *= (1.0 + title_matches * 1.5)

            if section.lower() in [
                "см. также", "см также", "источники информации", "источники", "литература"
            ]:
                final_score *= 0.5

            link_count = text.count("[[")
            text_length = len(text)
            if text_length > 0 and link_count > 3:
                link_ratio = (link_count * 50) / text_length
                if link_ratio > 0.3:
                    final_score *= 0.8

            text_lower = text.lower()
            if any(
                marker in text_lower
                for marker in ["определение", "теорема", "лемма", "доказательство", "утверждение", "алгоритм"]
            ):
                final_score *= 1.1

            enriched_results.append(
                {
                    "id": result.id,
                    "text": text,
                    "title": payload.get("title", ""),
                    "source": payload.get("source", ""),
                    "section": section,
                    "chunk_index": payload.get("chunk_index", 0),
                    "vector_score": vector_score,
                    "bm25_score": bm25_score,
                    "keyword_score": keyword_score,
                    "hybrid_score": hybrid_score,
                    "final_score": final_score,
                }
            )
        for r in enriched_results:
            if self.count_word_hits(r["text"], query_words) == 0:
                r["final_score"] *= 0.2

        enriched_results.sort(key=lambda x: x["final_score"], reverse=True)

        return enriched_results[:limit]




    def _normalize_text(self, text: str) -> str:
        return text.replace("_", " ").lower()

    def _extract_ngrams(self, text: str, n: int) -> set:
        normalized = self._normalize_text(text)
        words = re.findall(r"\b\w+\b", normalized)
        if len(words) < n:
            return set()
        return set(" ".join(words[i: i + n]) for i in range(len(words) - n + 1))

    def _clean_query(self, query_lower: str) -> str:
        return re.sub(
            r"^(что такое|как|где|когда|почему|какой|какая|какие)\s+", "", query_lower
        ).strip()

    def _exact_substring_score(self,
                               clean_query: str,
                               text_lower: str,
                               title_lower: str,
                               source_lower: str) -> float:
        score = 0.0
        if clean_query and clean_query in text_lower:
            score += 0.7
        if clean_query and clean_query in title_lower:
            score += 0.5
        if clean_query and clean_query in source_lower:
            score += 0.8
        return score



    def _ngram_match_score(
            self,
            query_for_ngrams: str,
            text_lower: str,
            title_lower: str,
            source_lower: str,
    ) -> float:
        score = 0.0
        query_bigrams = self._extract_ngrams(query_for_ngrams, 2)
        query_trigrams = self._extract_ngrams(query_for_ngrams, 3)
        text_bigrams = self._extract_ngrams(text_lower, 2)
        text_trigrams = self._extract_ngrams(text_lower, 3)
        title_bigrams = self._extract_ngrams(title_lower, 2)
        title_trigrams = self._extract_ngrams(title_lower, 3)
        source_bigrams = self._extract_ngrams(source_lower, 2)
        source_trigrams = self._extract_ngrams(source_lower, 3)

        if query_trigrams:
            trigram_text_matches = len(query_trigrams & text_trigrams)
            trigram_title_matches = len(query_trigrams & title_trigrams)
            trigram_source_matches = len(query_trigrams & source_trigrams)

            if trigram_text_matches > 0:
                score += 0.5 * (trigram_text_matches / len(query_trigrams))
            if trigram_title_matches > 0:
                score += 0.4 * (trigram_title_matches / len(query_trigrams))
            if trigram_source_matches > 0:
                score += 0.6 * (trigram_source_matches / len(query_trigrams))

        if query_bigrams:
            bigram_text_matches = len(query_bigrams & text_bigrams)
            bigram_title_matches = len(query_bigrams & title_bigrams)
            bigram_source_matches = len(query_bigrams & source_bigrams)

            if bigram_text_matches > 0:
                score += 0.3 * (bigram_text_matches / len(query_bigrams))
            if bigram_title_matches > 0:
                score += 0.25 * (bigram_title_matches / len(query_bigrams))
            if bigram_source_matches > 0:
                score += 0.4 * (bigram_source_matches / len(query_bigrams))

        return score

    def _filter_query_words(self, query_lower: str) -> Set[str]:
        stopwords = {
            "что", "такое", "это", "как", "где", "когда", "почему", "какой", "какая", "какие", "является",
        }
        words = set()
        for word in re.findall(r"\b\w+\b", query_lower):
            if word not in stopwords and len(word) > 1:
                words.add(word)
        return words

    def _word_match_score(
            self,
            query_words: Set[str],
            text_lower: str,
            title_lower: str,
            source_lower: str,
    ) -> float:
        if not query_words:
            return 0.0
        text_words = set(re.findall(r"\b\w+\b", text_lower))
        title_words = set(re.findall(r"\b\w+\b", title_lower))
        source_words = set(re.findall(r"\b\w+\b", source_lower))

        score = 0.0
        title_matches = len(query_words & title_words)
        if title_matches > 0:
            score += 0.2 * (title_matches / len(query_words))

        text_matches = len(query_words & text_words)
        if text_matches > 0:
            score += 0.15 * (text_matches / len(query_words))

        source_matches = len(query_words & source_words)
        if source_matches > 0:
            score += 0.25 * (source_matches / len(query_words))

        return score

    def _compute_keyword_score(self, query: str, payload: Dict) -> float:
        query_lower = self._normalize_text(query)
        text_lower = self._normalize_text(payload.get("text", ""))
        title_lower = self._normalize_text(payload.get("title", ""))
        source_lower = self._normalize_text(payload.get("source", ""))

        score = 0.0
        clean_query = self._clean_query(query_lower)
        score += self._exact_substring_score(clean_query, text_lower, title_lower, source_lower)
        query_for_ngrams = clean_query if clean_query else query_lower
        score += self._ngram_match_score(query_for_ngrams, text_lower, title_lower, source_lower)
        query_words = self._filter_query_words(query_lower)
        if not query_words:
            return min(score, 1.0)
        score += self._word_match_score(query_words, text_lower, title_lower, source_lower)
        return min(score, 1.0)

    def format_results(self, results: List[Dict], max_text_length: int = 500) -> str:
        if not results:
            return "Ничего не найдено."
        output = []
        output.append(f"Найдено результатов: {len(results)}\n")
        for i, result in enumerate(results, 1):
            output.append(f"{'=' * 60}")
            output.append(f"Результат #{i} (релевантность: {result['final_score']:.3f})")
            output.append(f"Заголовок: {result['title']}")
            if result["section"]:
                output.append(f"Секция: {result['section']}")
            output.append(f"Источник: {result['source']}")
            output.append(f"Vec:{result['vector_score']:.3f}|BM25:{result['bm25_score']:.3f}|Key:{result['keyword_score']:.3f}")
            output.append("")
            text = result["text"]
            output.append(text if len(text) <= max_text_length else text[:max_text_length] + "...") # Ограничить размер вывода
            output.append("")
        return "\n".join(output)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Поиск по LaTeX документам (гибридный)")
    parser.add_argument("query", type=str, help="Поисковый запрос")
    parser.add_argument("--host", type=str, default=os.getenv("QDRANT_HOST", "localhost"), help="Хост Qdrant")
    parser.add_argument("--port", type=int, default=int(os.getenv("QDRANT_PORT", "6333")), help="Порт Qdrant")
    parser.add_argument("--collection", type=str, default="latex_books", help="Название коллекции")
    parser.add_argument("--limit", type=int, default=10, help="Количество результатов")
    parser.add_argument("--model", type=str, default=os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"), help="Модель для embeddings")
    parser.add_argument("--bm25-index", type=str, default="/app/data/bm25_index.pkl", help="Путь к BM25 индексу")
    parser.add_argument("--alpha", type=float, default=0.3, help="Вес вектор-поиска в гибридном поиске (0.5 = равный вес вектора и BM25)")

    args = parser.parse_args()

    logger.info("Инициализация поисковика...")
    searcher = Searcher(
        qdrant_host=args.host,
        qdrant_port=args.port,
        collection_name=args.collection,
        embedding_model=args.model,
        bm25_index_path=args.bm25_index,
        alpha=args.alpha,
    )

    logger.info(f"\nПоиск: '{args.query}'")
    logger.info(f"{'=' * 60}\n")

    results = searcher.search(args.query, limit=args.limit)
    formatted = searcher.format_results(results)
    logger.info(formatted)

if __name__ == "__main__":
    main()
