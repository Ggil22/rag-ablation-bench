import numpy as np
import pytest

from rag_ablation_bench.retrieval import BM25, RetrieverFactory, rrf
from rag_ablation_bench.retrieval.base import top_k
from rag_ablation_bench.retrieval.text import chunk_text, tokenize
from rag_ablation_bench.schema import Chunk, Span


def make_chunks(*texts: str, headers: tuple[str, ...] = ()) -> list[Chunk]:
    headers = headers or tuple(f"DOC > Artículo {i}" for i in range(len(texts)))
    return [
        Chunk(f"c{i}", "d", t, h, (Span(f"s{i}", 0, len(t)),))
        for i, (t, h) in enumerate(zip(texts, headers, strict=True))
    ]


def test_tokenize_stems_folds_accents_and_drops_stopwords():
    assert tokenize("El período de prueba") == tokenize("el periodo de PRUEBAS")
    assert "de" not in tokenize("periodo de prueba", stem=False)
    assert tokenize("despidos", stem=False) != tokenize("despido", stem=False)
    assert tokenize("despidos") == tokenize("despido")
    assert "no" in tokenize("no podrá exceder")  # negations carry meaning


def test_chunk_text_header():
    c = make_chunks("texto")[0]
    assert chunk_text(c, header=True) == "DOC > Artículo 0\ntexto"
    assert chunk_text(c, header=False) == "texto"


def test_bm25_ranks_matching_chunk_first():
    chunks = make_chunks(
        "La fianza será de una mensualidad de renta.",
        "El periodo de prueba no podrá exceder de seis meses.",
        "Las vacaciones anuales no serán inferiores a treinta días naturales.",
    )
    bm25 = BM25(header=False)
    bm25.index(chunks)
    [[(best, _), *_]] = bm25.search(["¿Cuánto dura el período de prueba?"], k=3)
    assert best == 1


def test_bm25_rare_terms_weigh_more():
    chunks = make_chunks("plazo plazo plazo", "plazo recurso", "plazo", "plazo alzada")
    bm25 = BM25(header=False)
    bm25.index(chunks)
    scores = bm25.scores("plazo alzada")
    assert scores.argmax() == 3


def test_bm25_header_is_searchable():
    chunks = make_chunks(
        "texto genérico", "texto genérico", headers=("LAU > Fianza", "ET > Jornada")
    )
    with_header, without = BM25(header=True), BM25(header=False)
    with_header.index(chunks)
    without.index(chunks)
    assert with_header.search(["fianza"], 1)[0][0][0] == 0
    assert without.scores("fianza").max() == 0


def test_unknown_query_terms_score_zero():
    bm25 = BM25()
    bm25.index(make_chunks("uno", "dos"))
    assert not bm25.scores("inexistente").any()


def test_top_k_is_sorted_and_deterministic():
    scores = np.array([0.1, 0.9, 0.5, 0.9])
    assert top_k(scores, 3) == [(1, 0.9), (3, 0.9), (2, 0.5)]
    assert len(top_k(scores, 10)) == 4


def test_rrf():
    fused = rrf([[(0, 9.0), (1, 5.0)], [(1, 0.8), (2, 0.7)]], k=60)
    assert [i for i, _ in fused] == [1, 0, 2]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_factory_shares_instances_and_validates():
    factory = RetrieverFactory(
        {
            "bm25": {"type": "bm25"},
            "hyb": {"type": "hybrid", "components": ["bm25", "bm25"]},
            "bad": {"type": "nope"},
        }
    )
    assert factory("hyb").components[0] is factory("bm25")
    with pytest.raises(KeyError):
        factory("missing")
    with pytest.raises(ValueError):
        factory("bad")


def test_hybrid_search_runs():
    factory = RetrieverFactory(
        {
            "a": {"type": "bm25"},
            "b": {"type": "bm25", "stem": False},
            "h": {"type": "hybrid", "components": ["a", "b"]},
        }
    )
    h = factory("h")
    h.index(make_chunks("fianza de vivienda", "jornada de trabajo", "fianza de local"))
    ranking = h.search(["fianza"], k=2)[0]
    assert {i for i, _ in ranking} == {0, 2}
