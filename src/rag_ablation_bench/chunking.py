"""Chunking strategies. Sizes are measured in whitespace-separated words.

Every strategy works on (section_id, start, end) pieces and turns them into Chunks whose
text is copied verbatim from the sections, so each chunk knows exactly which characters of
which sections it covers.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Callable, Iterator

from rag_ablation_bench.schema import Chunk, Document, Section, Span

Piece = tuple[str, int, int]  # section_id, start, end

NUMBERED_SUBSECTION = re.compile(r"^\d+(?:\s*(?:bis|ter|quater))?\.\s")
SENTENCE = re.compile(r"\S.*?(?:[.;:](?=\s|$)|$)", re.DOTALL)


def _matches(pattern: str | re.Pattern, sec: Section, start: int = 0, end: int | None = None):
    end = len(sec.text) if end is None else end
    for m in re.finditer(pattern, sec.text[start:end]):
        yield sec.id, start + m.start(), start + m.end()


def _n_words(piece: Piece, sections: dict[str, Section]) -> int:
    sid, start, end = piece
    return len(sections[sid].text[start:end].split())


def _to_chunk(pieces: list[Piece], doc: Document, sections: dict[str, Section], n: int) -> Chunk:
    """Merge consecutive pieces of the same section into one span and copy their text."""
    spans: list[Span] = []
    for sid, start, end in pieces:
        if spans and spans[-1].section_id == sid:
            spans[-1] = Span(sid, spans[-1].start, end)
        else:
            spans.append(Span(sid, start, end))
    first = sections[spans[0].section_id]
    return Chunk(
        id=f"{doc.id}:{n}",
        doc_id=doc.id,
        text="\n\n".join(sections[s.section_id].text[s.start : s.end] for s in spans),
        header=" > ".join((doc.short_title, *first.path, first.title)),
        spans=tuple(spans),
    )


def _split(piece: Piece, sections: dict[str, Section], size: int) -> list[Piece]:
    """Recursively break a piece over `size` words: paragraphs, then sentences, then words."""
    if _n_words(piece, sections) <= size:
        return [piece]
    sid, start, end = piece
    sec = sections[sid]
    for pattern in (r"[^\n]+", SENTENCE):
        parts = list(_matches(pattern, sec, start, end))
        if len(parts) > 1:
            return [p for part in parts for p in _split(part, sections, size)]
    words = list(_matches(r"\S+", sec, start, end))
    return [
        (sid, words[i][1], words[min(i + size, len(words)) - 1][2])
        for i in range(0, len(words), size)
    ]


def _pack(pieces: list[Piece], sections: dict[str, Section], size: int) -> Iterator[list[Piece]]:
    """Greedily group consecutive pieces while the group stays within `size` words."""
    group: list[Piece] = []
    total = 0
    for piece in pieces:
        n = _n_words(piece, sections)
        if group and total + n > size:
            yield group
            group, total = [], 0
        group.append(piece)
        total += n
    if group:
        yield group


# --- strategies --------------------------------------------------------------------------


def fixed(doc: Document, size: int, overlap: int = 0) -> list[list[Piece]]:
    """Sliding window of `size` words over the whole document, ignoring its structure."""
    if not 0 <= overlap < size:
        raise ValueError("overlap must be in [0, size)")
    words = [w for sec in doc.sections for w in _matches(r"\S+", sec)]
    groups = []
    i = 0
    while i < len(words):
        groups.append(words[i : i + size])
        if i + size >= len(words):
            break
        i += size - overlap
    return groups


def recursive(doc: Document, size: int) -> list[list[Piece]]:
    """Paragraph/sentence splitting packed up to `size`; may merge neighbouring sections."""
    sections = {s.id: s for s in doc.sections}
    pieces = [p for sec in doc.sections for p in _split((sec.id, 0, len(sec.text)), sections, size)]
    return list(_pack(pieces, sections, size))


def structure(doc: Document, size: int) -> list[list[Piece]]:
    """One chunk per section; long sections split at numbered sub-sections ("1.", "2.")."""
    sections = {s.id: s for s in doc.sections}
    groups = []
    for sec in doc.sections:
        lines = list(_matches(r"[^\n]+", sec))
        units: list[Piece] = []  # each numbered sub-section with its trailing paragraphs
        for i, (sid, start, end) in enumerate(lines):
            # The title line always joins the first unit: alone it would be a useless chunk.
            if i == 1 or (units and not NUMBERED_SUBSECTION.match(sec.text[start:end])):
                units[-1] = (sid, units[-1][1], end)
            else:
                units.append((sid, start, end))
        pieces = [p for unit in units for p in _split(unit, sections, size)]
        groups.extend(_pack(pieces, sections, size))
    return groups


STRATEGIES: dict[str, Callable[..., list[list[Piece]]]] = {
    "fixed": fixed,
    "recursive": recursive,
    "structure": structure,
}


def chunk_documents(docs: list[Document], strategy: str, **params) -> list[Chunk]:
    chunks = []
    for doc in docs:
        sections = {s.id: s for s in doc.sections}
        for n, group in enumerate(STRATEGIES[strategy](doc, **params)):
            chunks.append(_to_chunk(group, doc, sections, n))
    return chunks


def chunk_stats(chunks: list[Chunk]) -> dict[str, float]:
    sizes = sorted(len(c.text.split()) for c in chunks)
    return {
        "chunks": len(chunks),
        "mean_words": statistics.fmean(sizes),
        "p50_words": sizes[len(sizes) // 2],
        "p95_words": sizes[int(len(sizes) * 0.95)],
        "max_words": sizes[-1],
        "multi_section": sum(len(c.section_ids) > 1 for c in chunks) / len(chunks),
    }
