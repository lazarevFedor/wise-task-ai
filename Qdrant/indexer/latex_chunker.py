import os
import re
from pathlib import Path
from typing import List, Dict
from pylatexenc.latex2text import LatexNodes2Text


class LaTeXChunker:

    def __init__(
        self,
        chunk_size: int | None = None,
        overlap: int | None = None,
        min_chunk_size: int | None = None,
    ):
        self.chunk_size = chunk_size or int(os.getenv("CHUNK_MAX_LEN", "2000"))
        self.overlap = overlap or int(os.getenv("CHUNK_OVERLAP", "300"))
        self.min_chunk_size = min_chunk_size or int(os.getenv("CHUNK_MIN_LEN", "1500"))
        self.converter = LatexNodes2Text()
        print(
            f"Chunker init: chunk_size={self.chunk_size},"
            f" overlap={self.overlap}, "
            f"min={self.min_chunk_size}"
        )

    def read_latex_file(self, filepath: Path) -> tuple[str, str, str]:
        raw_latex = ""
        for encoding in ["utf-8", "cp1251", "koi8-r"]:
            try:
                raw_latex = filepath.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue

        if not raw_latex:
            raw_latex = filepath.read_bytes().decode("utf-8", errors="ignore")

        try:
            plain_text = self.converter.latex_to_text(raw_latex)
        except Exception:
            plain_text = raw_latex

        title = self._extract_title_from_filename(filepath.name)

        title_match = re.search(r"==\s*([^=]+?)\s*==", plain_text)
        if title_match:
            title = title_match.group(1).strip()

        return raw_latex, plain_text, title

    def _extract_title_from_filename(self, filename: str) -> str:
        name = filename.replace(".tex", "")
        name = name.replace("Просмотр_исходного_текста_страницы_", "")
        name = name.replace("_", " ")
        return name.strip()

    def chunk_document(self, filepath: Path) -> List[Dict]:
        raw_latex, plain_text, title = self.read_latex_file(filepath)

        sections = self._extract_sections(plain_text)

        chunks = []
        chunk_id = 0

        for section in sections:
            section_chunks = self._split_section(section, title, filepath.name)

            for chunk_text in section_chunks:
                text_without_prefix = (
                    chunk_text.split("\n\n", 1)[-1]
                    if "\n\n" in chunk_text
                    else chunk_text
                )
                if len(text_without_prefix.strip()) < self.min_chunk_size:
                    continue
                chunks.append(
                    {
                        "id": chunk_id,
                        "text": chunk_text,
                        "title": self._extract_title_from_filename(filepath.name),
                        "source": self._extract_title_from_filename(filepath.name),
                        "chunk_index": chunk_id,
                        "section": section.get("title", ""),
                    }
                )
                chunk_id += 1

        return chunks

    def _extract_sections(self, text: str) -> List[Dict]:
        sections = []

        parts = re.split(r"={2,}\s*([^=]+?)\s*={2,}", text)

        if len(parts) == 1:
            sections.append({"title": "", "content": text.strip()})
        else:
            current_title = ""
            for i, part in enumerate(parts):
                if i == 0 and part.strip():
                    sections.append({"title": "", "content": part.strip()})
                elif i % 2 == 1:
                    current_title = part.strip()
                else:
                    if part.strip():
                        sections.append(
                            {"title": current_title, "content": part.strip()}
                        )

        return sections

    from typing import Dict, List

    def _format_prefix(self, source: str, section_title: str) -> str:
        prefix = source
        if section_title:
            prefix += f" | {section_title}"
        return prefix

    def _chunk_large_paragraph(self, para: str, prefix: str) -> List[str]:
        chunks = []
        for i in range(0, len(para), self.chunk_size - self.overlap):
            chunk_text = para[i: i + self.chunk_size]
            chunks.append(f"{prefix}\n\n{chunk_text}")
        return chunks

    def _flush_current_chunk(
        self, chunks: List[str], current_chunk: List[str], prefix: str
    ):
        if current_chunk:
            chunks.append(f"{prefix}\n\n" + "\n\n".join(current_chunk))

    def _process_small_paragraph(
        self,
        para: str,
        current_chunk: List[str],
        current_size: int,
        chunks: List[str],
        prefix: str,
    ) -> (List[str], int):
        para_size = len(para)

        if current_size + para_size + 2 > self.chunk_size and current_chunk:
            self._flush_current_chunk(chunks, current_chunk, prefix)

            if len(current_chunk) > 0:
                new_chunk = [current_chunk[-1], para]
                new_size = len(current_chunk[-1]) + para_size + 2
            else:
                new_chunk = [para]
                new_size = para_size

            return new_chunk, new_size

        current_chunk.append(para)
        current_size += para_size + 2
        return current_chunk, current_size

    def _split_section(self, section: Dict, title: str, source: str) -> List[str]:
        content = section["content"]
        section_title = section["title"]

        prefix = self._format_prefix(source, section_title)

        if len(content) <= self.chunk_size:
            return [f"{prefix}\n\n{content}"]

        paragraphs = content.split("\n\n")

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_size = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_size = len(para)

            if para_size > self.chunk_size:
                self._flush_current_chunk(chunks, current_chunk, prefix)
                current_chunk = []
                current_size = 0

                large_parts = self._chunk_large_paragraph(para, prefix)
                chunks.extend(large_parts)
                continue

            current_chunk, current_size = self._process_small_paragraph(
                para, current_chunk, current_size, chunks, prefix
            )

        self._flush_current_chunk(chunks, current_chunk, prefix)

        return chunks


def chunk_latex_file(filepath: Path, chunk_size: int = 2000) -> List[Dict]:
    chunker = LaTeXChunker(chunk_size=chunk_size)
    return chunker.chunk_document(filepath)
