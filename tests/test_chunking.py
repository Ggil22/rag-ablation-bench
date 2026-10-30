import random
import re

import pytest

from rag_ablation_bench.chunking import chunk_documents, chunk_stats, fixed
from rag_ablation_bench.schema import Chunk, Document, Section, Span, load_chunks, save_chunks


def make_section(rng: random.Random, doc_id: str, i: int) -> Section:
    """An article with numbered sub-sections, long paragraphs and an unpunctuated run."""
    lines = [f"Artículo {i}. Título del artículo {i}."]
    for n in range(1, rng.randint(1, 5)):
        lines.append(f"{n}. " + " ".join(f"palabra{k}." for k in range(rng.randint(3, 120))))
        for _ in range(rng.randint(0, 2)):
            lines.append(" ".join(f"texto{k}" for k in range(rng.randint(1, 80))) + ".")
    if rng.random() < 0.2:  # a paragraph with no sentence boundary at all
        lines.append(" ".join("x" for _ in range(rng.randint(100, 300))))
    return Section(
        id=f"{doc_id}#a{i}",
        doc_id=doc_id,
        kind="article",
        title=lines[0],
        path=("TÍTULO I. Generales",),
        text="\n".join(lines),
    )


def make_doc(seed: int) -> Document:
    rng = random.Random(seed)
    doc_id = f"DOC-{seed}"
    return Document(
        id=doc_id,
        corpus="test",
        title="Documento de prueba",
        short_title="DP",
        url="",
        sections=tuple(make_section(rng, doc_id, i) for i in range(1, rng.randint(2, 12))),
    )


DOCS = [make_doc(seed) for seed in range(20)]
CONFIGS = [
    ("fixed", {"size": 50, "overlap": 10}),
    ("fixed", {"size": 64, "overlap": 0}),
    ("recursive", {"size": 50}),
    ("recursive", {"size": 200}),
    ("structure", {"size": 50}),
    ("structure", {"size": 200}),
]


def word_positions(doc: Document) -> set[tuple[str, int]]:
    return {(s.id, m.start()) for s in doc.sections for m in re.finditer(r"\S+", s.text)}


@pytest.mark.parametrize(("strategy", "params"), CONFIGS)
def test_chunk_invariants(strategy, params):
    for doc in DOCS:
        sections = {s.id: s for s in doc.sections}
        chunks = chunk_documents([doc], strategy, **params)
        covered = set()
        for chunk in chunks:
            # The text is exactly the concatenation of the spans it claims to cover.
            texts = [sections[s.section_id].text[s.start : s.end] for s in chunk.spans]
            assert chunk.text == "\n\n".join(texts)
            for span in chunk.spans:
                assert 0 <= span.start < span.end <= len(sections[span.section_id].text)
                text = sections[span.section_id].text
                covered |= {
                    (span.section_id, span.start + m.start())
                    for m in re.finditer(r"\S+", text[span.start : span.end])
                }
            assert len(chunk.text.split()) <= params["size"]
            assert chunk.header.startswith("DP > TÍTULO I. Generales > Artículo")
        assert covered == word_positions(doc), "some words are in no chunk"
        if strategy == "structure":
            assert all(len(c.section_ids) == 1 for c in chunks)


def test_fixed_windows_overlap():
    doc = DOCS[0]
    groups = fixed(doc, size=40, overlap=8)
    for a, b in zip(groups, groups[1:], strict=False):
        assert a[-8:] == b[:8]
    assert all(len(g) == 40 for g in groups[:-1])


def test_fixed_rejects_bad_overlap():
    with pytest.raises(ValueError):
        fixed(DOCS[0], size=10, overlap=10)


def test_structure_keeps_short_sections_whole():
    doc = DOCS[3]
    chunks = chunk_documents([doc], "structure", size=10_000)
    assert [c.text for c in chunks] == [s.text for s in doc.sections]


def test_structure_splits_at_numbered_subsections():
    text = "Artículo 1. Título.\n1. Uno uno uno.\nSigue uno.\n2. Dos dos dos.\n3. Tres tres."
    sec = Section("D#a1", "D", "article", "Artículo 1. Título.", (), text)
    doc = Document("D", "test", "t", "D", "", (sec,))
    chunks = chunk_documents([doc], "structure", size=10)
    assert [c.text for c in chunks] == [
        "Artículo 1. Título.\n1. Uno uno uno.\nSigue uno.",
        "2. Dos dos dos.\n3. Tres tres.",
    ]
    # The title line stays with sub-section 1 instead of becoming a chunk on its own.
    assert all(c.header == "D > Artículo 1. Título." for c in chunks)


def test_span_overlap():
    assert Span("a", 0, 10).overlaps(Span("a", 9, 20))
    assert not Span("a", 0, 10).overlaps(Span("a", 10, 20))
    assert not Span("a", 0, 10).overlaps(Span("b", 0, 10))


def test_chunks_roundtrip_and_stats(tmp_path):
    chunks = chunk_documents(DOCS[:3], "recursive", size=50)
    path = tmp_path / "chunks.jsonl"
    save_chunks(chunks, path)
    assert load_chunks(path) == chunks
    assert chunk_stats(chunks)["chunks"] == len(chunks)
    assert isinstance(chunks[0], Chunk)
