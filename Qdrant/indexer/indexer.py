import os
import time
 import pickle
from pathlib import Path
from typing import Dict
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import re

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding

from latex_chunker import LaTeXChunker

import logging
logger = logging.getLogger(__name__)



class SimpleIndexer:

    def __init__(
            self,
            qdrant_host: str = "localhost",
            qdrant_port: int = 6333,
            collection_name: str = "latex_books",
            embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            bm25_index_path: str = "/app/data/bm25_index.pkl",
    ):
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection_name = collection_name
        self.bm25_index_path = bm25_index_path

        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)

        logger.info(f"Загружаем модель: {embedding_model}")
        self.embedding_model = TextEmbedding(embedding_model)

        self.vector_size = 768
        logger.info(f"Размерность векторов: {self.vector_size}")

        self.chunker = LaTeXChunker()

        self.bm25 = None
        self.corpus = []

    def _tokenize_russian(self, text: str):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return tokens

    def wait_for_qdrant(self, timeout: int = 120):
        logger.info(f"Ожидание Qdrant на {self.qdrant_host}:{self.qdrant_port}...")
        start = time.time()

        while time.time() - start < timeout:
            try:
                self.client.get_collections()
                logger.info("Qdrant готов!")
                return
            except Exception:
                time.sleep(2)

        raise TimeoutError(f"Qdrant не отвечает после {timeout}s")

    def create_collection(self, recreate: bool = False):
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)

        if collection_exists:
            if recreate:
                logger.info(f"Удаляем существующую коллекцию: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
            else:
                logger.info(f"Коллекция {self.collection_name} уже существует")
                return

        logger.info(f"Создаем коллекцию: {self.collection_name}")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size, distance=Distance.COSINE
            ),
        )
        logger.info(f"Коллекция {self.collection_name} создана")

    def index_directory(self, directory: Path, max_files: int | None = None):
        if not directory.exists():
            raise FileNotFoundError(f"Директория не найдена: {directory}")

        tex_files = list(directory.glob("*.tex"))

        if max_files:
            tex_files = tex_files[:max_files]

        logger.info(f"Найдено {len(tex_files)} LaTeX файлов")

        if not tex_files:
            logger.error("Нет файлов для индексации")
            return

        all_points = []
        point_id = 0
        tokenized_corpus = []

        for filepath in tqdm(tex_files, desc="Обработка файлов"):
            try:
                chunks = self.chunker.chunk_document(filepath)

                texts = [chunk["text"] for chunk in chunks]
                embeddings = list(self.embedding_model.embed(texts))

                for chunk, embedding in zip(chunks, embeddings):
                    point = PointStruct(
                        id=point_id,
                        vector=list(embedding),
                        payload={
                            "text": chunk["text"],
                            "title": chunk["title"],
                            "source": chunk["source"],
                            "chunk_index": chunk["chunk_index"],
                            "section": chunk["section"],
                        },
                    )
                    all_points.append(point)


                    self.corpus.append(chunk["text"])
                    tokenized_corpus.append(
                        self._tokenize_russian(chunk["text"])
                    )

                    point_id += 1

            except Exception as e:
                logger.error(f"\n Ошибка обработки {filepath.name}: {e}")
                continue

        batch_size = 100
        logger.info(f"\nЗагрузка {len(all_points)} чанков в Qdrant...")

        for i in tqdm(range(0, len(all_points), batch_size), desc="Загрузка в Qdrant"):
            batch = all_points[i: i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name, points=batch, wait=True
            )

        logger.info(f"\n Загружено {len(all_points)} чанков из {len(tex_files)} файлов")


        logger.info(f"\nИндексируем {len(tokenized_corpus)} документов в BM25...")
        self.bm25 = BM25Okapi(tokenized_corpus)

        logger.info(f"Сохраняем BM25 индекс в {self.bm25_index_path}...")
        with open(self.bm25_index_path, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "corpus": self.corpus,
            }, f)
        logger.info(f"BM25 индекс сохранён")

    def load_bm25_index(self):
        if os.path.exists(self.bm25_index_path):
            logger.info(f"Загружаем BM25 индекс из {self.bm25_index_path}...")
            with open(self.bm25_index_path, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.corpus = data["corpus"]
            logger.info(f"BM25 индекс загружен ({len(self.corpus)} документов)")
            return True
        else:
            logger.error(f"BM25 индекс не найден по пути {self.bm25_index_path}")
            return False

    def get_collection_info(self) -> Dict:
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
            }
        except Exception as e:
            return {"error": str(e)}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Индексация LaTeX документов в Qdrant")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/data/latex_books",
        help="Директория с LaTeX файлами",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("QDRANT_HOST", "localhost"),
        help="Хост Qdrant",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("QDRANT_PORT", "6333")),
        help="Порт Qdrant",
    )
    parser.add_argument(
        "--collection", type=str, default="latex_books", help="Название коллекции"
    )
    parser.add_argument("--recreate", action="store_true", help="Пересоздать коллекцию")
    parser.add_argument(
        "--max-files", type=int, default=None, help="Максимальное количество файлов"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        ),
        help="Модель для embeddings",
    )
    parser.add_argument(
        "--bm25-index",
        type=str,
        default="./bm25_index.pkl",
        help="Путь к BM25 индексу",
    )

    args = parser.parse_args()

    indexer = SimpleIndexer(
        qdrant_host=args.host,
        qdrant_port=args.port,
        collection_name=args.collection,
        embedding_model=args.model,
        bm25_index_path=args.bm25_index,
    )

    indexer.wait_for_qdrant()

    indexer.create_collection(recreate=args.recreate)

    data_dir = Path(args.data_dir)
    indexer.index_directory(data_dir, max_files=args.max_files)

    info = indexer.get_collection_info()
    logger.info(f"\n{'=' * 50}")
    logger.info("Статистика коллекции:")
    logger.info(f"  Название: {info.get('name', 'N/A')}")
    logger.info(f"  Количество точек: {info.get('points_count', 'N/A')}")
    logger.info(f"  Статус: {info.get('status', 'N/A')}")
    logger.info(f"{'=' * 50}")


if __name__ == "__main__":
    main()
