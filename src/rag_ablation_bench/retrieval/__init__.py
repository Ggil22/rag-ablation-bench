"""Retrievers and the factory that builds them from `configs/retrieval.yaml`."""

from __future__ import annotations

from rag_ablation_bench.retrieval.base import Ranking, Retriever
from rag_ablation_bench.retrieval.bm25 import BM25
from rag_ablation_bench.retrieval.fusion import Hybrid, rrf

__all__ = ["BM25", "Hybrid", "Ranking", "Retriever", "RetrieverFactory", "rrf"]


class RetrieverFactory:
    """Builds named retrievers from config, sharing instances so that a hybrid and its dense
    component reuse one loaded model instead of putting two copies on the GPU."""

    def __init__(self, configs: dict[str, dict], llm=None):
        """`llm` maps a model name from configs/models.yaml to an LLM client; it is only
        needed by query-transform retrievers."""
        self.configs = configs
        self.llm = llm
        self._built: dict[str, Retriever] = {}

    def __call__(self, name: str) -> Retriever:
        if name not in self._built:
            self._built[name] = self._build(name)
        return self._built[name]

    def _build(self, name: str) -> Retriever:
        if name not in self.configs:
            raise KeyError(f"Unknown retriever {name!r}")
        params = dict(self.configs[name])
        kind = params.pop("type")
        params.pop("llm", None)  # marker: calls an API, excluded from the default grid
        if kind == "bm25":
            return BM25(name, **params)
        if kind == "dense":
            from rag_ablation_bench.retrieval.dense import Dense

            return Dense(name, **params)
        if kind == "hybrid":
            return Hybrid(name, components=[self(c) for c in params.pop("components")], **params)
        if kind == "rerank":
            from rag_ablation_bench.retrieval.rerank import Reranked

            return Reranked(name, base=self(params.pop("base")), **params)
        if kind == "query":
            from rag_ablation_bench.retrieval.query import QueryTransform

            if self.llm is None:
                raise ValueError(f"{name!r} needs an LLM but the factory has none")
            llm = self.llm(params.pop("model"))
            return QueryTransform(name, base=self(params.pop("base")), llm=llm, **params)
        raise ValueError(f"Unknown retriever type {kind!r} for {name!r}")
