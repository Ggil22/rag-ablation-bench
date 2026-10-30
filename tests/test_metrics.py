import pytest

from rag_ablation_bench.chunking import SEPARATOR, chunk_documents
from rag_ablation_bench.evaluation.dataset import Fact
from rag_ablation_bench.evaluation.metrics import (
    coverage,
    facts_found,
    is_relevant,
    score_ranking,
    truncate,
    within_budget,
)
from rag_ablation_bench.evaluation.stats import bootstrap_ci, paired_diff_ci
from rag_ablation_bench.schema import Chunk, Span
from tests.test_chunking import DOCS


def chunk(*spans: tuple[str, int, int], text: str = "") -> Chunk:
    return Chunk("c", "d", text or "x " * 10, "h", tuple(Span(*s) for s in spans))


def test_coverage_unions_chunks():
    gold = Span("s", 0, 100)
    assert coverage(gold, [chunk(("s", 0, 40))]) == 0.4
    assert coverage(gold, [chunk(("s", 0, 40)), chunk(("s", 30, 70))]) == 0.7
    assert coverage(gold, [chunk(("t", 0, 100))]) == 0


def test_fact_split_across_two_chunks_is_found_when_both_retrieved():
    fact = Fact((Span("s", 50, 150),))
    left, right = chunk(("s", 0, 100)), chunk(("s", 100, 200))
    assert facts_found((fact,), [left, right]) == 1
    assert facts_found((fact,), [left]) == 1  # exactly half covered
    assert facts_found((fact,), [chunk(("s", 0, 90))]) == 0


def test_fact_alternatives():
    fact = Fact((Span("a", 0, 10), Span("b", 0, 10)))
    assert facts_found((fact,), [chunk(("b", 0, 10))]) == 1


def test_small_chunk_inside_long_evidence_is_relevant():
    facts = (Fact((Span("s", 0, 1000),)),)
    assert is_relevant(chunk(("s", 400, 500)), facts)
    assert not is_relevant(chunk(("s", 990, 1100)), facts)  # 10 of 110 chars


def test_truncate_keeps_text_and_spans_consistent():
    sections = {s.id: s for d in DOCS for s in d.sections}
    for c in chunk_documents(DOCS[:5], "recursive", size=80):
        for n in (1, 5, 17, 79, 500):
            t = truncate(c, n)
            assert len(t.text.split()) == min(n, len(c.text.split()))
            assert t.text == SEPARATOR.join(
                sections[s.section_id].text[s.start : s.end] for s in t.spans
            )


def test_budget_is_exact_for_every_chunk_size():
    for size in (50, 128, 300):
        chunks = chunk_documents(DOCS[:3], "fixed", size=size, overlap=0)
        total = sum(len(c.text.split()) for c in chunks)
        for budget in (100, 250, 1000):
            got = sum(len(c.text.split()) for c in within_budget(chunks, budget))
            assert got == min(budget, total)


def test_score_ranking():
    facts = (Fact((Span("s", 0, 10),)), Fact((Span("t", 0, 10),)))
    ranked = [chunk(("x", 0, 10)), chunk(("s", 0, 10)), chunk(("t", 0, 10))]
    m = score_ranking(ranked, facts)
    assert m["recall@1"] == 0 and m["recall@3"] == 1
    assert m["complete@1"] == 0 and m["complete@3"] == 1
    assert m["mrr@10"] == pytest.approx(1 / 2)


def test_bootstrap_ci():
    assert bootstrap_ci([0.5] * 20) == (0.5, 0.5, 0.5)
    mean, lo, hi = bootstrap_ci([0, 1] * 20)
    assert lo < mean == 0.5 < hi


def test_paired_diff_detects_consistent_improvement():
    a = [1.0] * 15 + [0.0] * 5
    b = [0.0] * 15 + [0.0] * 5
    mean, lo, hi = paired_diff_ci(a, b)
    assert mean == 0.75 and lo > 0
    with pytest.raises(ValueError):
        paired_diff_ci([1.0], [1.0, 2.0])
