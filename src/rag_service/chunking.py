from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class DocumentChunk:
    doc_id: str
    chunk_id: str
    title: str
    language: str
    text: str
    metadata: dict


def parse_front_matter(text: str) -> Tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, parts[2].strip()


def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            if len(para) <= chunk_size:
                current = para
            else:
                start = 0
                while start < len(para):
                    chunks.append(para[start : start + chunk_size])
                    start += max(1, chunk_size - overlap)
                current = ""
    if current:
        chunks.append(current)
    return chunks


def iter_source_files(corpus_path: Path) -> Iterable[Path]:
    for pattern in ("*.md", "*.txt"):
        yield from sorted(corpus_path.glob(pattern))


def load_chunks(corpus_path: Path, chunk_size: int = 850, overlap: int = 120) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    for file_path in iter_source_files(corpus_path):
        text = file_path.read_text(encoding="utf-8")
        metadata, body = parse_front_matter(text)
        doc_id = metadata.get("doc_id") or file_path.stem
        title = metadata.get("title") or file_path.stem.replace("_", " ").title()
        language = metadata.get("language", "unknown")
        for idx, chunk in enumerate(split_text(body, chunk_size=chunk_size, overlap=overlap), start=1):
            chunks.append(
                DocumentChunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}::chunk_{idx:03d}",
                    title=title,
                    language=language,
                    text=chunk,
                    metadata={**metadata, "source_file": str(file_path.name)},
                )
            )
    return chunks
