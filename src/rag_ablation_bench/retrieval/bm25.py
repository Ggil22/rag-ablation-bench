"""Okapi BM25 over an inverted index, written out so every scoring choice is visible."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from rag_ablation_bench.retrieval.base import Ranking, top_k
from rag_ablation_bench.retrieval.text import chunk_text, tokenize
from rag_ablation_bench.schema import Chunk


class BM25:
    def __init__(
        self,
        name: str = "bm25",
        *,
        k1: float = 1.2,
        b: float = 0.75,
        stem: bool = True,
        stopwords: bool = True,
        header: bool = True,
    ):
        self.name = name
        self.k1, self.b = k1, b
        self.stem, self.stopwords, self.header = stem, stopwords, header

    def _tokens(self, text: str) -> list[str]:
        return tokenize(text, stem=self.stem, stopwords=self.stopwords)

    def index(self, chunks: list[Chunk]) -> None:
        docs = [self._tokens(chunk_text(c, header=self.header)) for c in chunks]
        self.n_docs = len(docs)
        lengths = np.array([len(d) for d in docs], dtype=np.float64)
        # Per-document length normalisation of the term-frequency saturation.
        self._norm = self.k1 * (1 - self.b + self.b * lengths / max(lengths.mean(), 1e-9))

        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_id, tokens in enumerate(docs):
            for term, tf in Counter(tokens).items():
                postings[term].append((doc_id, tf))
        self._postings = {
            term: (np.array([d for d, _ in p]), np.array([tf for _, tf in p], dtype=np.float64))
            for term, p in postings.items()
        }
        # Lucene's IDF variant, which is always positive.
        self._idf = {
            term: math.log(1 + (self.n_docs - len(p[0]) + 0.5) / (len(p[0]) + 0.5))
            for term, p in self._postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(self.n_docs)
        for term, qtf in Counter(self._tokens(query)).items():
            if term not in self._postings:
                continue
            docs, tf = self._postings[term]
            scores[docs] += qtf * self._idf[term] * tf * (self.k1 + 1) / (tf + self._norm[docs])
        return scores

    def search(self, queries: list[str], k: int) -> list[Ranking]:
        return [top_k(self.scores(q), k) for q in queries]
