"""Dense retrieval with sentence-transformers embedding models (needs the `models` extra)."""

from __future__ import annotations

import hashlib
from functools import cache
from pathlib import Path

import numpy as np

from rag_ablation_bench.retrieval.base import Ranking, top_k
from rag_ablation_bench.retrieval.text import chunk_text
from rag_ablation_bench.schema import Chunk


@cache  # one copy per model per process, however many retrievers use it
def load_device_model(loader, model: str):
    """Load a sentence-transformers model on the GPU in fp16 when there is one."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    return loader(model, device=device, model_kwargs={"torch_dtype": dtype})


class Dense:
    def __init__(
        self,
        name: str,
        *,
        model: str,
        query_prefix: str = "",
        doc_prefix: str = "",
        max_seq_length: int = 1024,
        batch_size: int = 16,
        header: bool = True,
        cache_dir: Path | None = Path(".cache/embeddings"),
    ):
        self.name, self.model_name = name, model
        self.query_prefix, self.doc_prefix = query_prefix, doc_prefix
        self.max_seq_length, self.batch_size = max_seq_length, batch_size
        self.header, self.cache_dir = header, cache_dir
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = load_device_model(SentenceTransformer, self.model_name)
            self._model.max_seq_length = self.max_seq_length
        return self._model

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        return self.model.encode(
            [prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 500,
        ).astype(np.float32)

    def _cache_path(self, texts: list[str]) -> Path | None:
        if self.cache_dir is None:
            return None
        h = hashlib.sha256()
        for part in (self.model_name, self.doc_prefix, str(self.max_seq_length), *texts):
            h.update(part.encode("utf-8") + b"\0")
        return self.cache_dir / self.model_name.replace("/", "__") / f"{h.hexdigest()[:24]}.npy"

    def index(self, chunks: list[Chunk]) -> None:
        texts = [chunk_text(c, header=self.header) for c in chunks]
        path = self._cache_path(texts)
        if path is not None and path.exists():
            self._emb = np.load(path)
            return
        self._emb = self._encode(texts, self.doc_prefix)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, self._emb)

    def search(self, queries: list[str], k: int) -> list[Ranking]:
        q = self._encode(queries, self.query_prefix)
        return [top_k(scores, k) for scores in q @ self._emb.T]
