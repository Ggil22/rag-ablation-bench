"""Second-stage reranking with a cross-encoder (needs the `models` extra)."""

from __future__ import annotations

from rag_ablation_bench.retrieval.base import Ranking, Retriever
from rag_ablation_bench.retrieval.dense import load_device_model
from rag_ablation_bench.retrieval.text import chunk_text
from rag_ablation_bench.schema import Chunk


class Reranked:
    """Take the top `candidates` of a base retriever and reorder them with a cross-encoder,
    which reads query and chunk together instead of comparing separate embeddings."""

    def __init__(
        self,
        name: str,
        *,
        base: Retriever,
        model: str,
        candidates: int = 50,
        max_length: int = 1024,
        batch_size: int = 16,
        header: bool = True,
    ):
        self.name, self.base, self.model_name = name, base, model
        self.candidates, self.max_length, self.batch_size = candidates, max_length, batch_size
        self.header = header
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = load_device_model(CrossEncoder, self.model_name)
            self._model.max_length = self.max_length
        return self._model

    def index(self, chunks: list[Chunk]) -> None:
        self.base.index(chunks)
        self._texts = [chunk_text(c, header=self.header) for c in chunks]

    def search(self, queries: list[str], k: int) -> list[Ranking]:
        results = []
        for query, ranking in zip(queries, self.base.search(queries, self.candidates), strict=True):
            ids = [idx for idx, _ in ranking]
            scores = self.model.predict(
                [(query, self._texts[i]) for i in ids],
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            reranked = sorted(zip(ids, map(float, scores), strict=True), key=lambda x: -x[1])
            results.append(reranked[:k])
        return results
