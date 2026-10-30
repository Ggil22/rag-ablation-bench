import pytest

from rag_ablation_bench.evaluation.dataset import InvalidQuestions, load_questions, resolve_ref
from tests.test_boe import parse

DOCS = [parse()]


def write(tmp_path, text: str):
    path = tmp_path / "q.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_resolve_ref_by_title_and_block_id():
    assert resolve_ref("LP: Artículo 1", DOCS).id == "BOE-A-2000-1#a1"
    assert resolve_ref("LP#a2", DOCS).id == "BOE-A-2000-1#a2"
    assert resolve_ref("LP: Disposición adicional primera", DOCS).kind == "provision"
    with pytest.raises(KeyError):
        resolve_ref("XX: Artículo 1", DOCS)
    with pytest.raises(KeyError):
        resolve_ref("LP: Artículo 9", DOCS)


def test_quotes_become_spans(tmp_path):
    path = write(
        tmp_path,
        """
- id: q1
  question: ¿Plazo?
  answer: Quince días.
  type: lookup
  evidence:
    - ref: "LP: Artículo 1"
      quote: El plazo es de quince días.
      alternatives:
        - ref: "LP#a2"
          quote: "Hasta 12.450 | 9,50"
- id: q2
  question: ¿IVA?
  answer: UNANSWERABLE
  type: unanswerable
""",
    )
    q1, q2 = load_questions(path, DOCS)
    a1 = next(s for s in DOCS[0].sections if s.id.endswith("#a1"))
    span = q1.facts[0].spans[0]
    assert a1.text[span.start : span.end] == "El plazo es de quince días."
    assert len(q1.facts[0].spans) == 2
    assert q1.answerable and not q2.answerable


def test_all_errors_reported_at_once(tmp_path):
    path = write(
        tmp_path,
        """
- id: q1
  question: a
  answer: b
  type: lookup
  evidence:
    - ref: "LP: Artículo 1"
      quote: texto que no existe
- id: q1
  question: a
  answer: b
  type: lookup
  evidence:
    - ref: "LP: Artículo 99"
      quote: x
""",
    )
    with pytest.raises(InvalidQuestions) as exc:
        load_questions(path, DOCS)
    message = str(exc.value)
    assert "quote not found" in message
    assert "duplicate id" in message
    assert "matches 0 sections" in message


def test_seed_set_matches_real_corpus_when_built():
    """The real seed set is validated against the real corpus when it exists locally."""
    from pathlib import Path

    from rag_ablation_bench.schema import load_documents

    corpus = Path("data/processed/boe/documents.jsonl")
    if not corpus.exists():
        pytest.skip("corpus not built (run `rab corpus boe`)")
    questions = load_questions(Path("evalsets/boe/seed.yaml"), load_documents(corpus))
    assert len({q.id for q in questions}) == len(questions)
