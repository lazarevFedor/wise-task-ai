#!/usr/bin/env python3
"""
Industrial RAG Indexer: комбинирует структурное и семантическое разбиение
для сохранения математических определений и формул.
"""

import os
import time
import argparse
import re
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Tuple
from dataclasses import dataclass
import uuid

from tqdm import tqdm
from pylatexenc.latex2text import LatexNodes2Text
from embedding import EmbeddingModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct


@dataclass
class ChunkConfig:
    """Конфигурация разбиения на чанки"""
    max_chunk_size: int = 3000
    min_chunk_size: int = 200
    overlap_size: int = 100
    structural_priority: bool = True


class IndustrialChunker:
    """Промышленный чанкер с комбинированной стратегией"""
    
    def __init__(self, config: ChunkConfig):
        self.config = config
        # Wiki-like patterns kept for legacy converted sources
        self.structural_patterns = [
            (r'\{\{Определение[^}]*\}(.*?)\}\}', 'definition'),
            (r'\{\{Теорема[^}]*\}(.*?)\}\}', 'theorem'),
            (r'\{\{Утверждение[^}]*\}(.*?)\}\}', 'statement'),
            (r'\{\{Доказательство[^}]*\}(.*?)\}\}', 'proof'),
            (r'== [^=]+ ==', 'section'),
            (r'=== [^=]+ ===', 'subsection'),
        ]
    
    def chunk_text(self, text: str) -> List[Dict]:
        """Основной метод разбиения текста на чанки"""
        
        # Стратегия 1: Структурное разбиение (приоритет для математики)
        if self.config.structural_priority:
            structural_chunks = self._structural_chunking(text)
            if structural_chunks and self._validate_chunks(structural_chunks):
                return structural_chunks
        
        # Стратегия 2: Семантическое разбиение с перекрытием
        semantic_chunks = self._semantic_chunking(text)
        return semantic_chunks

    def chunk_latex(self, raw_latex: str, plain_text: str) -> List[Dict]:
        """LaTeX-осознанное разбиение: пытается выделять окружения и секции из сырого LaTeX.
        При неудаче — откатывается к разбиению plain_text.
        """
        try:
            chunks: list[Dict] = []
            # 1) Секции
            section_pattern = re.compile(r'\\(section|chapter)\*?\{([^}]*)\}', flags=re.IGNORECASE)
            subsection_pattern = re.compile(r'\\(subsection|subsubsection)\*?\{([^}]*)\}', flags=re.IGNORECASE)
            # 2) Теоремные окружения
            env_names = ['definition', 'theorem', 'lemma', 'proposition', 'corollary', 'proof']
            env_regex = '|'.join(env_names)
            env_pattern = re.compile(r'\\begin\{(' + env_regex + r')\}([\s\S]*?)\\end\{\1\}', flags=re.IGNORECASE)
            # 3) Формулы-блоки
            eq_patterns = [
                re.compile(r'\\begin\{equation\}([\s\S]*?)\\end\{equation\}', flags=re.IGNORECASE),
                re.compile(r'\\\[([\s\S]*?)\\\]', flags=re.IGNORECASE),
                re.compile(r'\$\$([\s\S]*?)\$\$', flags=re.IGNORECASE),
            ]

            markers: list[Tuple[int, int, str, str]] = []
            for m in section_pattern.finditer(raw_latex):
                markers.append((m.start(), m.end(), 'section', m.group(0)))
            for m in subsection_pattern.finditer(raw_latex):
                markers.append((m.start(), m.end(), 'subsection', m.group(0)))
            for m in env_pattern.finditer(raw_latex):
                env = m.group(1).lower()
                markers.append((m.start(), m.end(), env, m.group(0)))
            for pat in eq_patterns:
                for m in pat.finditer(raw_latex):
                    markers.append((m.start(), m.end(), 'equation', m.group(0)))

            if not markers:
                # fallback to plain text strategy
                return self.chunk_text(plain_text)

            markers.sort(key=lambda t: t[0])

            last_end = 0
            # Накапливаем фрагменты; длинные режем
            for start, end, kind, content in markers:
                if start > last_end:
                    preceding = raw_latex[last_end:start].strip()
                    if len(preceding) >= self.config.min_chunk_size:
                        chunks.extend(self._split_large_content(preceding, 'content'))
                # сам структурный фрагмент
                if len(content) <= self.config.max_chunk_size:
                    chunks.append({'content': content, 'type': kind})
                else:
                    chunks.extend(self._split_large_content(content, kind))
                last_end = end
            if last_end < len(raw_latex):
                tail = raw_latex[last_end:].strip()
                if len(tail) >= self.config.min_chunk_size:
                    chunks.extend(self._split_large_content(tail, 'content'))

            # Валидация; если плохо — используем plain_text
            if not self._validate_chunks(chunks):
                return self.chunk_text(plain_text)
            return chunks
        except Exception:
            return self.chunk_text(plain_text)
    
    def _structural_chunking(self, text: str) -> List[Dict]:
        """Разбиение по структурным элементам"""
        chunks = []
        last_end = 0
        
        # Собираем все структурные элементы
        matches = []
        for pattern, chunk_type in self.structural_patterns:
            for match in re.finditer(pattern, text, re.DOTALL):
                matches.append((match.start(), match.end(), chunk_type, match.group(0)))
        
        # Сортируем по позициям
        matches.sort(key=lambda x: x[0])
        
        for start, end, chunk_type, content in matches:
            # Добавляем текст перед структурным элементом
            if start > last_end:
                preceding = text[last_end:start].strip()
                if len(preceding) >= self.config.min_chunk_size:
                    chunks.extend(self._split_large_content(preceding, 'content'))
            
            # Добавляем структурный элемент
            if len(content) <= self.config.max_chunk_size:
                chunks.append({'content': content, 'type': chunk_type})
            else:
                # Если структурный элемент слишком большой, разбиваем его
                chunks.extend(self._split_large_content(content, chunk_type))
            
            last_end = end
        
        # Добавляем оставшийся текст
        if last_end < len(text):
            remaining = text[last_end:].strip()
            if len(remaining) >= self.config.min_chunk_size:
                chunks.extend(self._split_large_content(remaining, 'content'))
        
        return chunks
    
    def _semantic_chunking(self, text: str) -> List[Dict]:
        """Семантическое разбиение на предложения с перекрытием"""
        # Разбиваем на предложения
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Если предложение слишком длинное, разбиваем его
            if len(sentence) > self.config.max_chunk_size:
                if current_chunk:
                    chunks.append({'content': current_chunk, 'type': 'content'})
                    current_chunk = ""
                
                # Разбиваем длинное предложение
                parts = self._split_long_sentence(sentence)
                chunks.extend([{'content': part, 'type': 'content'} for part in parts])
                continue
            
            # Проверяем, помещается ли предложение в текущий чанк
            if current_chunk and len(current_chunk) + len(sentence) > self.config.max_chunk_size:
                chunks.append({'content': current_chunk, 'type': 'content'})
                # Добавляем перекрытие
                overlap_start = max(0, len(current_chunk) - self.config.overlap_size)
                current_chunk = current_chunk[overlap_start:] + " " + sentence

            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append({'content': current_chunk, 'type': 'content'})
        
        return chunks
    
    def _split_large_content(self, content: str, chunk_type: str) -> List[Dict]:
        """Разбивает большой контент на части"""
        # 🚀 Не резать важные блоки
        if chunk_type in ('definition', 'theorem', 'lemma', 'proposition', 'corollary', 'proof'):
            return [{'content': content, 'type': chunk_type}]
        
        if len(content) <= self.config.max_chunk_size:
            return [{'content': content, 'type': chunk_type}]
        
        if chunk_type != 'content':
            sub_chunks = self._split_by_logical_breaks(content, chunk_type)
            if sub_chunks:
                return sub_chunks
        
        return self._split_fixed_size(content, chunk_type)



    
    def _split_by_logical_breaks(self, content: str, chunk_type: str) -> List[Dict]:
        """Разбивает по логическим разрывам (точки, запятые в формулах)"""
        chunks = []
        
        # Ищем точки с запятыми и точки как потенциальные места разрыва
        break_positions = []
        for match in re.finditer(r'[.;]', content):
            # Проверяем, что это не часть формулы
            if not self._is_in_math_context(content, match.start()):
                break_positions.append(match.start())
        
        if not break_positions:
            return []
        
        # Создаем чанки по найденным позициям
        start = 0
        for pos in break_positions:
            chunk_content = content[start:pos + 1].strip()
            if (len(chunk_content) >= self.config.min_chunk_size and 
                len(chunk_content) <= self.config.max_chunk_size):
                chunks.append({'content': chunk_content, 'type': chunk_type})
                start = pos + 1
                # если маленький chunk_type=='content', объединяем с предыдущим
        if chunks and chunk_type == 'content' and len(chunks[-1]['content']) < self.config.min_chunk_size:
            chunks[-1]['content'] += " " + chunk_content
        else:
            chunks.append({'content': chunk_content, 'type': chunk_type})

        # Оставшаяся часть
        if start < len(content):
            remaining = content[start:].strip()
            if len(remaining) >= self.config.min_chunk_size:
                chunks.append({'content': remaining, 'type': chunk_type})
        
        return chunks
    
    def _split_fixed_size(self, content: str, chunk_type: str) -> List[Dict]:
        """Фиксированное разбиение на чанки с перекрытием"""
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + self.config.max_chunk_size
            if end >= len(content):
                chunk_content = content[start:].strip()
                if len(chunk_content) >= self.config.min_chunk_size:
                    chunks.append({'content': chunk_content, 'type': chunk_type})
                break
            
            # Ищем хорошее место для разрыва (пробел или конец предложения)
            break_pos = self._find_break_position(content, end)
            chunk_content = content[start:break_pos].strip()
            
            if len(chunk_content) >= self.config.min_chunk_size:
                chunks.append({'content': chunk_content, 'type': chunk_type})
            
            start = break_pos - self.config.overlap_size
        
        return chunks
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Разбивает очень длинное предложение"""
        parts = []
        start = 0
        
        while start < len(sentence):
            end = start + self.config.max_chunk_size
            if end >= len(sentence):
                parts.append(sentence[start:].strip())
                break
            
            # Ищем место для разрыва
            break_pos = self._find_break_position(sentence, end)
            parts.append(sentence[start:break_pos].strip())
            start = break_pos
        
        return parts
    
    def _find_break_position(self, text: str, suggested_pos: int) -> int:
        """Находит оптимальную позицию для разрыва текста"""
        # Сначала ищем конец предложения
        for pos in range(suggested_pos, max(0, suggested_pos - 100), -1):
            if pos < len(text) and text[pos] in '.!?':
                return pos + 1
        
        # Затем ищем пробел
        for pos in range(suggested_pos, max(0, suggested_pos - 50), -1):
            if pos < len(text) and text[pos].isspace():
                return pos + 1
        
        # Fallback: используем suggested_pos
        return min(suggested_pos, len(text))
    
    def _is_in_math_context(self, text: str, position: int) -> bool:
        """Проверяет, находится ли позиция в математическом контексте"""
        # Простая эвристика: проверяем окружение символов
        left_context = text[max(0, position-10):position]
        right_context = text[position:min(len(text), position+10)]
        
        # Если вокруг цифры, буквы или математических символов - вероятно формула
        math_indicators = r'[0-9a-zA-Z\(\)\[\]\+\-\*/=]'
        return (re.search(math_indicators, left_context) and 
                re.search(math_indicators, right_context))
    
    def _validate_chunks(self, chunks: List[Dict]) -> bool:
        """Проверяет качество чанков"""
        if not chunks:
            return False
        
        # Проверяем, что нет слишком больших чанков
        for chunk in chunks:
            if len(chunk['content']) > self.config.max_chunk_size * 1.5:
                return False
        
        return True


def wait_for_qdrant(host: str, port: int, timeout: int = 120):
    """Ожидание готовности Qdrant"""
    client = QdrantClient(host=host, port=port)
    start = time.time()
    while True:
        try:
            client.get_collections()
            return
        except Exception:
            if time.time() - start > timeout:
                raise TimeoutError("Qdrant not ready after waiting")
            time.sleep(1)


def latex_to_text(path: Path) -> tuple[str, str, str]:
    """Конвертирует LaTeX в текст и извлекает заголовок; возвращает также сырой LaTeX/текст"""
    data = path.read_bytes()
    
    def _decode_attempt(b: bytes) -> str:
        for enc in ('utf-8', 'cp1251', 'koi8-r'):
            try:
                txt = b.decode(enc)
                if txt.count('�') > 5:
                    continue
                bad_pairs = txt.count('Ð') + txt.count('Ñ')
                if enc != 'utf-8' and bad_pairs < 5:
                    return txt
                if enc == 'utf-8' and bad_pairs > 50:
                    continue
                return txt
            except Exception:
                continue
        return b.decode('utf-8', errors='ignore')
    
    raw = _decode_attempt(data)
    suffix = path.suffix.lower()

    if suffix == '.txt':
        text = " ".join(raw.split())
        first_line = raw.splitlines()[0].strip() if raw.splitlines() else path.stem
        title = first_line[:100]
        return text, title, raw

    converter = LatexNodes2Text()
    try:
        text = converter.latex_to_text(raw)
    except Exception:
        text = raw
    
    text = " ".join(text.split())

    # Извлекаем заголовок
    title = ""
    try:
        m = re.search(r"\\(?:section|chapter|title)\*?\{([^}]+)\}", raw)
        if m:
            title = m.group(1).strip()
        else:
            first_line = text.split('\n', 1)[0]
            title = first_line.strip()[:100]
    except Exception:
        title = ""

    return text, title, raw


# Лемматизатор
try:
    import pymorphy2
    _lemmatizer = pymorphy2.MorphAnalyzer()
except Exception:
    _lemmatizer = None

_WORD_RE = re.compile(r"[\w']+")


if _lemmatizer:
    @lru_cache(maxsize=100_000)
    def _normalize_word(word: str) -> str:
        try:
            return _lemmatizer.parse(word)[0].normal_form
        except Exception:
            return word
else:
    def _normalize_word(word: str) -> str:
        return word


def lemmatize_text(text: str) -> str:
    """Лемматизирует текст"""
    if not text:
        return ""
    words = _WORD_RE.findall(text.lower())
    if not _lemmatizer:
        return " ".join(words)
    lemmas = []
    for w in words:
        lemma = _normalize_word(w)
        lemmas.append(lemma)
    return " ".join(lemmas)


def unique_tokens(sequence: str) -> list[str]:
    """Удаляет дубликаты токенов с сохранением порядка"""
    seen: set[str] = set()
    ordered: list[str] = []
    for token in sequence.split():
        if not token or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def prepare_records_from_file(
    file_path: Path,
    chunker: IndustrialChunker,
    start_id: int = 1,
) -> tuple[list[dict], int]:
    """Подготавливает записи из файла с промышленным чанкером"""
    text, title, raw = latex_to_text(file_path)
    
    # LaTeX-aware chunking first, fallback to semantic/plain
    chunks_data = chunker.chunk_latex(raw, text)
    
    def _sanitize_wiki(text: str) -> str:
        """Очищает вики-разметку, сохраняя математический контент"""
        if not text:
            return ''
        # Убираем только мешающие элементы, сохраняя математику
        text = re.sub(r"\[\[(?:Файл:|File:)[^\]]*\]\]", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
        text = re.sub(r"\{\{[^}]*\}\}", " ", text)
        text = re.sub(r"\|\-+", " ", text)
        text = re.sub(r"\|", " ", text)
        text = " ".join(text.split())
        return text.strip()

    title_clean = _sanitize_wiki(title)
    prefix = (title_clean + ' - ') if title_clean else ''
    title_norm = lemmatize_text(title_clean)
    title_lemmas = unique_tokens(title_norm)
    
    records: list[dict] = []
    for idx, chunk_data in enumerate(chunks_data):
        # Детерминированный UUID5 на основе имени источника и индекса чанка
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"qdrant::{file_path.name}::{idx}"))
        
        chunk_content = chunk_data['content']
        chunk_type = chunk_data['type']
        
        chunk_clean = _sanitize_wiki(chunk_content)
        text_with_title = prefix + chunk_clean
        
        text_norm = lemmatize_text(text_with_title)
        lemmas = unique_tokens(text_norm)
        
        payload = {
            "source": str(file_path.name),
            "chunk_index": idx,
            "title": title_clean,
            "text": chunk_clean,
            "chunk_text": chunk_clean,
            "text_norm": text_norm,
            "lemmas": lemmas,
            "title_lemmas": title_lemmas,
            "chunk_type": chunk_type,
            "chunk_length": len(chunk_clean)
        }
        
        records.append({
            "id": pid,
            "payload": payload,
            "embed_text": text_with_title,
        })
    
    next_id = start_id + len(records)
    return records, next_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/data/latex_books', help='Папка с .tex файлами')
    parser.add_argument('--recreate', action='store_true', help='Пересоздать коллекцию')
    parser.add_argument('--batch-size', type=int, default=32, help='Размер батча для эмбеддинга')
    parser.add_argument('--warmup', action='store_true', help='Прогрев модели перед индексацией')
    parser.add_argument('--max-files', type=int, default=None, help='Ограничить количество файлов')
    parser.add_argument('--extra-dir', default=os.environ.get('EXTRA_LATEX_DIR', '/data/latex_sources'))
    parser.add_argument('--chunk-max', type=int, default=None, help='Максимальный размер чанка в символах')
    parser.add_argument('--chunk-min', type=int, default=None, help='Минимальный размер чанка в символах')
    parser.add_argument('--chunk-overlap', type=int, default=None, help='Перекрытие чанков в символах при семантическом разбиении')
    parser.add_argument('--no-struct-priority', action='store_true', help='Отключить приоритет структурного разбиения')
    args = parser.parse_args()

    env_max = os.environ.get('CHUNK_MAX_LEN')
    env_min = os.environ.get('CHUNK_MIN_LEN')
    env_overlap = os.environ.get('CHUNK_OVERLAP')

    max_chunk_size = args.chunk_max if args.chunk_max is not None else (int(env_max) if env_max else 5000)
    min_chunk_size = args.chunk_min if args.chunk_min is not None else (int(env_min) if env_min else 200)
    overlap_size = args.chunk_overlap if args.chunk_overlap is not None else (int(env_overlap) if env_overlap else 200)
    structural_priority = not args.no_struct_priority

    if min_chunk_size > max_chunk_size:
        print(f"[warn] min_chunk_size {min_chunk_size} > max_chunk_size {max_chunk_size}, исправляю")
        min_chunk_size = max_chunk_size // 4
    if overlap_size >= max_chunk_size:
        overlap_size = max_chunk_size // 5

    print(f"Chunk config: max={max_chunk_size} min={min_chunk_size} overlap={overlap_size} structural_priority={structural_priority}")

    chunk_config = ChunkConfig(
        max_chunk_size=max_chunk_size,
        min_chunk_size=min_chunk_size,
        overlap_size=overlap_size,
        structural_priority=structural_priority,
    )

    chunker = IndustrialChunker(chunk_config)

    qdrant_host = os.environ.get('QDRANT_HOST', 'qdrant')
    qdrant_port = int(os.environ.get('QDRANT_PORT', '6333'))
    collection_name = os.environ.get('COLLECTION_NAME', 'latex_books')
    model_name = os.environ.get('EMBEDDING_MODEL', 'intfloat/multilingual-e5-small')

    print(f"Waiting for Qdrant at {qdrant_host}:{qdrant_port}...")
    wait_for_qdrant(qdrant_host, qdrant_port)

    print(f"Loading embedding model: {model_name}")
    model = EmbeddingModel(model_name)

    if args.warmup:
        try:
            _ = model.encode(['warmup'])
            print('Model warmup complete.')
        except Exception as e:
            print('Warmup failed:', e)
    
    dim = model.dimension

    def embed_fn(texts: List[str]) -> List[List[float]]:
        return [list(map(float, v)) for v in model.encode(texts)]

    if dim is None:
        raise RuntimeError("Failed to determine embedding dimension")

    client = QdrantClient(host=qdrant_host, port=qdrant_port, prefer_grpc=False)

    if args.recreate:
        print(f"(Re)creating collection '{collection_name}' with dim={dim}")
        try:
            client.recreate_collection(
                collection_name=collection_name, 
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )
        except Exception as e:
            print("Failed to recreate collection:", e)
            raise
    else:
        try:
            client.get_collection(collection_name)
            print(f"Collection '{collection_name}' exists")
        except Exception:
            print(f"Creating collection '{collection_name}'")
            client.recreate_collection(
                collection_name=collection_name, 
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data dir {data_dir} not found; nothing to index.")
        return

    candidates = []
    seen_paths: set[Path] = set()

    def collect_from(folder: Path):
        if not folder.exists():
            return
        exts = ('.tex', '.txt')
        for ext in exts:
            for file_path in folder.rglob(f'*{ext}'):
                try:
                    real = file_path.resolve()
                except Exception:
                    real = file_path
                if real in seen_paths:
                    continue
                seen_paths.add(real)
                candidates.append(file_path)

    collect_from(data_dir)
    extra_dir = Path(args.extra_dir) if args.extra_dir else None
    if extra_dir and extra_dir != data_dir:
        collect_from(extra_dir)

    if not candidates:
        print("No .tex files found in", data_dir)
        return

    candidates.sort()
    if args.max_files is not None:
        candidates = candidates[:args.max_files]

    total = 0
    next_id = 1
    embed_inputs: list[str] = []
    record_buffer: list[dict] = []

    def flush_batch():
        nonlocal total
        if not embed_inputs:
            return
        vectors = embed_fn(embed_inputs)
        points = [
            PointStruct(
                id=rec['id'],
                vector=list(map(float, vec)),
                payload=rec['payload'],
            )
            for rec, vec in zip(record_buffer, vectors)
        ]
        client.upsert(collection_name=collection_name, points=points, wait=True)
        total += len(points)
        embed_inputs.clear()
        record_buffer.clear()

    print(f"Processing {len(candidates)} files with industrial chunker...")
    for path in tqdm(candidates, desc='Indexing files'):
        records, next_id = prepare_records_from_file(path, chunker, start_id=next_id)
        for rec in records:
            embed_inputs.append(rec['embed_text'])
            record_buffer.append(rec)
            if len(embed_inputs) >= args.batch_size:
                flush_batch()

    flush_batch()

    print(f"Indexing complete. Total chunks: {total}")
    try:
        import requests as _rq
        r = _rq.get(f"http://{qdrant_host}:{qdrant_port}/collections/{collection_name}", timeout=5)
        r.raise_for_status()
        js = r.json()
        result = js.get('result', {})
        actual_count = result.get('points_count')
        print(f"Verification: Qdrant has {actual_count} points in '{collection_name}' (expected ~{total})")
    except Exception as e:
        print(f"Verification (HTTP) failed: {e}")


if __name__ == '__main__':
    main()