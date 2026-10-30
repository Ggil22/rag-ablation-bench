"""Retrieval metrics scored against gold evidence spans, independent of the chunking.

A fact is *found* when the retrieved chunks together cover at least `MIN_COVERAGE` of one
of its spans; evidence split across two adjacent chunks still counts if both are retrieved.
A chunk is *relevant* when it overlaps a gold span by at least `MIN_COVERAGE` of the shorter
of the two, so a 128-word chunk inside a long piece of evidence is relevant too.
"""

from __future__ import annotations

import re
from dataclasses import replace

from rag_ablation_bench.chunking import SEPARATOR
from rag_ablation_bench.evaluation.dataset import Fact
from rag_ablation_bench.schema import Chunk, Span

MIN_COVERAGE = 0.5
KS = (1, 3, 5, 10, 20)
BUDGETS = (250, 500, 1000, 2000)  # words of context handed to the generator


def _overlap(a: Span, b: Span) -> int:
    if a.section_id != b.section_id:
        return 0
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def coverage(gold: Span, chunks: list[Chunk]) -> float:
    """Fraction of the gold span's characters covered by the union of the chunks."""
    intervals = sorted(
        (max(s.start, gold.start), min(s.end, gold.end))
        for c in chunks
        for s in c.spans
        if _overlap(s, gold)
    )
    covered, end = 0, gold.start
    for lo, hi in intervals:
        lo = max(lo, end)
        if hi > lo:
            covered += hi - lo
            end = hi
    return covered / (gold.end - gold.start)


def is_relevant(chunk: Chunk, facts: tuple[Fact, ...]) -> bool:
    for fact in facts:
        for gold in fact.spans:
            for s in chunk.spans:
                ov = _overlap(s, gold)
                if ov and ov >= MIN_COVERAGE * min(s.end - s.start, gold.end - gold.start):
                    return True
    return False


def facts_found(facts: tuple[Fact, ...], chunks: list[Chunk]) -> int:
    return sum(any(coverage(gold, chunks) >= MIN_COVERAGE for gold in fact.spans) for fact in facts)


def truncate(chunk: Chunk, n_words: int) -> Chunk:
    """The chunk cut after its first `n_words` words, with its spans cut to match."""
    words = list(re.finditer(r"\S+", chunk.text))
    if n_words >= len(words):
        return chunk
    cut = words[n_words - 1].end() if n_words > 0 else 0
    spans, offset = [], 0
    for s in chunk.spans:  # chunk text is the span texts joined by SEPARATOR
        if cut <= offset:
            break
        spans.append(Span(s.section_id, s.start, s.start + min(s.end - s.start, cut - offset)))
        offset += s.end - s.start + len(SEPARATOR)
    return replace(chunk, text=chunk.text[:cut], spans=tuple(spans))


def within_budget(ranked: list[Chunk], budget: int) -> list[Chunk]:
    """The first `budget` words of context: top chunks in order, the last one truncated.
    Every chunk size then gets exactly the same amount of text."""
    out, used = [], 0
    for chunk in ranked:
        if used >= budget:
            break
        n = len(chunk.text.split())
        out.append(chunk if used + n <= budget else truncate(chunk, budget - used))
        used += n
    return out


def score_ranking(ranked: list[Chunk], facts: tuple[Fact, ...]) -> dict[str, float]:
    """All metrics for one question. `ranked` must hold at least max(KS) chunks if available."""
    n = len(facts)
    m: dict[str, float] = {}
    for k in KS:
        found = facts_found(facts, ranked[:k])
        m[f"recall@{k}"] = found / n
        m[f"complete@{k}"] = float(found == n)  # every fact the answer needs is in context
    for budget in BUDGETS:
        m[f"recall@{budget}w"] = facts_found(facts, within_budget(ranked, budget)) / n
    first = next((i for i, c in enumerate(ranked[:10], 1) if is_relevant(c, facts)), None)
    m["mrr@10"] = 1 / first if first else 0.0
    return m
