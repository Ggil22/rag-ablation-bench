"""Core data model shared by every corpus, chunker and retriever.

A corpus is a list of Documents, each split into Sections: the natural units of the source
(an article of a law, a section of a paper). Sections have stable ids, and evaluation gold
labels point at character spans inside them. Chunks record which section spans they cover,
so the same gold labels score every chunking strategy.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Section:
    id: str  # globally unique and stable, e.g. "BOE-A-2015-11430#a37"
    doc_id: str
    kind: str  # article | provision | preamble | annex | ...
    title: str
    path: tuple[str, ...]  # enclosing headings, outermost first
    text: str  # full text including the title line; paragraphs separated by "\n"
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Document:
    id: str
    corpus: str
    title: str
    short_title: str
    url: str
    sections: tuple[Section, ...]
    metadata: dict = field(default_factory=dict)

    def word_count(self) -> int:
        return sum(len(s.text.split()) for s in self.sections)


@dataclass(frozen=True)
class Span:
    section_id: str
    start: int  # character offsets into Section.text
    end: int

    def overlaps(self, other: Span) -> bool:
        return (
            self.section_id == other.section_id
            and self.start < other.end
            and other.start < self.end
        )


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    text: str
    header: str  # where the chunk lives: "ET > TÍTULO I ... > Artículo 37. ..."
    spans: tuple[Span, ...]

    @property
    def section_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(s.section_id for s in self.spans))


# --- JSONL persistence -------------------------------------------------------------------


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def save_documents(docs: Iterable[Document], path: Path) -> None:
    _write_jsonl(path, (asdict(d) for d in docs))


def load_documents(path: Path) -> list[Document]:
    docs = []
    for row in _read_jsonl(path):
        sections = tuple(Section(**{**s, "path": tuple(s["path"])}) for s in row.pop("sections"))
        docs.append(Document(**row, sections=sections))
    return docs


def save_chunks(chunks: Iterable[Chunk], path: Path) -> None:
    _write_jsonl(path, (asdict(c) for c in chunks))


def load_chunks(path: Path) -> list[Chunk]:
    return [
        Chunk(**{**row, "spans": tuple(Span(**s) for s in row["spans"])})
        for row in _read_jsonl(path)
    ]
