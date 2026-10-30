"""The retriever interface: index chunks once, then rank them for a batch of queries."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from rag_ablation_bench.schema import Chunk

Ranking = list[tuple[int, float]]  # (chunk index, score), best first


class Retriever(Protocol):
    name: str

    def index(self, chunks: list[Chunk]) -> None: ...

    def search(self, queries: list[str], k: int) -> list[Ranking]: ...


def top_k(scores: np.ndarray, k: int) -> Ranking:
    """Indices of the k highest scores, sorted, with ties broken by index for determinism."""
    k = min(k, len(scores))
    candidates = np.argpartition(-scores, k - 1)[:k]
    order = sorted(candidates, key=lambda i: (-scores[i], i))
    return [(int(i), float(scores[i])) for i in order]
