"""Hybrid retrieval by reciprocal rank fusion (Cormack et al., 2009)."""

from __future__ import annotations

from collections import defaultdict

from rag_ablation_bench.retrieval.base import Ranking, Retriever
from rag_ablation_bench.schema import Chunk


def rrf(rankings: list[Ranking], k: int = 60) -> Ranking:
    """Fuse rankings by sum of 1 / (k + rank). Uses ranks only, so scores on different
    scales (BM25 vs cosine) need no calibration."""
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (idx, _) in enumerate(ranking, start=1):
            fused[idx] += 1 / (k + rank)
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))


class Hybrid:
    def __init__(self, name: str, *, components: list[Retriever], depth: int = 100, k: int = 60):
        self.name, self.components, self.depth, self.k = name, components, depth, k

    def index(self, chunks: list[Chunk]) -> None:
        for r in self.components:
            r.index(chunks)

    def search(self, queries: list[str], k: int) -> list[Ranking]:
        per_retriever = [r.search(queries, self.depth) for r in self.components]
        return [
            rrf([rankings[i] for rankings in per_retriever], self.k)[:k]
            for i in range(len(queries))
        ]
