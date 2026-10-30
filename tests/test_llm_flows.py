"""Generation, judging, question generation, review and labeling with a scripted LLM."""

import json
from collections.abc import Callable

import pytest
import yaml

from rag_ablation_bench.chunking import chunk_documents
from rag_ablation_bench.evaluation import e2e
from rag_ablation_bench.evaluation.dataset import Fact, Question
from rag_ablation_bench.evaluation.judge import Judge, deterministic_outcome
from rag_ablation_bench.evaluation.labeling import agreement, cohen_kappa, label, load_labels
from rag_ablation_bench.evaluation.question_gen import find_quote, generate, lexical_overlap
from rag_ablation_bench.evaluation.review import review
from rag_ablation_bench.llm import LLMResponse, ModelConfig, QuotaExhausted, _is_daily_quota
from rag_ablation_bench.rag import RAG
from rag_ablation_bench.retrieval import BM25
from rag_ablation_bench.retrieval.query import QueryTransform
from rag_ablation_bench.schema import Span
from tests.test_boe import parse

DOC = parse()
SECTIONS = {s.id.split("#")[1]: s for s in DOC.sections}
CHUNKS = chunk_documents([DOC], "structure", size=512)


class FakeLLM:
    """Answers with `reply(system, user) -> dict` and records every prompt."""

    def __init__(self, reply: Callable[[str, str], dict]):
        self.reply = reply
        self.prompts: list[str] = []
        self.cfg = ModelConfig("fake", "fake", "fake-model", "", None, price_in=1.0, price_out=2.0)

    def ask_json(self, system: str, user: str, **kwargs):
        self.prompts.append(user)
        data = self.reply(system, user)
        if data is None:
            raise ValueError("unparseable")
        return data, LLMResponse(
            json.dumps(data), prompt_tokens=100, completion_tokens=10, latency_s=0.5
        )


def question(qid="q1", answerable=True) -> Question:
    a1 = SECTIONS["a1"]
    quote = "El plazo es de quince días."
    start = a1.text.index(quote)
    facts = (Fact((Span(a1.id, start, start + len(quote)),)),) if answerable else ()
    return Question(
        qid,
        "¿Cuál es el plazo?",
        "Quince días." if answerable else "UNANSWERABLE",
        "lookup" if answerable else "unanswerable",
        facts,
    )


# --- LLMResponse.json ------------------------------------------------------------------------


def test_response_json_tolerates_fences_and_text():
    assert LLMResponse('```json\n{"a": 1}\n```').json() == {"a": 1}
    assert LLMResponse('Aquí está: {"a": {"b": 2}} fin').json() == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        LLMResponse("sin json").json()


def test_model_cost():
    cfg = ModelConfig("m", "p", "m", "", None, price_in=0.5, price_out=2.0)
    assert cfg.cost(1_000_000, 500_000) == pytest.approx(1.5)
    assert ModelConfig("m", "p", "m", "", None).cost(10, 10) is None


# --- RAG -------------------------------------------------------------------------------------


def make_rag(reply) -> tuple[RAG, FakeLLM]:
    bm25 = BM25()
    bm25.index(CHUNKS)
    llm = FakeLLM(reply)
    return RAG(bm25, CHUNKS, llm, budget=200), llm


def test_rag_answer_with_citations():
    rag, llm = make_rag(
        lambda s, u: {"answer": "Quince días.", "citations": [1, 99, "x"], "abstain": False}
    )
    a = rag.answer("¿plazo en días?")
    assert a.answer == "Quince días." and not a.abstain
    assert a.citations == [0]  # out-of-range and non-numeric citations dropped
    assert "[1] LP >" in llm.prompts[0]
    assert sum(len(c.text.split()) for c in a.context) <= 200


def test_rag_unparseable_output_is_an_error():
    rag, _ = make_rag(lambda s, u: None)
    assert rag.answer("¿plazo?").error


# --- Judge -----------------------------------------------------------------------------------


def judge_reply(verdict: str, supported: bool):
    def reply(system, user):
        if "Respuesta de referencia" in user:
            return {"reason": "r", "verdict": verdict}
        return {"reason": "r", "supported": supported}

    return reply


def test_deterministic_outcomes():
    kw = {"answer": "x", "error": None}
    assert deterministic_outcome(answerable=False, abstain=True, **kw) == "correct_abstention"
    assert deterministic_outcome(answerable=False, abstain=False, **kw) == "hallucinated"
    assert deterministic_outcome(answerable=True, abstain=True, **kw) == "false_abstention"
    assert deterministic_outcome(answerable=True, abstain=False, **kw) is None
    assert deterministic_outcome(answerable=True, abstain=False, answer="", error="boom") == "error"


def test_judge_verdicts():
    judge = Judge(FakeLLM(judge_reply("correct", True)))
    assert judge.correctness("q", "ref", "ans")[0] == "correct"
    assert judge.faithfulness("ans", ["frag"]) is True
    assert Judge(FakeLLM(judge_reply("nonsense", True))).correctness("q", "r", "a")[0] == "error"
    assert Judge(FakeLLM(lambda s, u: None)).faithfulness("a", ["f"]) is None


# --- End to end --------------------------------------------------------------------------------


def answering_rag():
    return make_rag(lambda s, u: {"answer": "Quince días.", "citations": [1], "abstain": False})[0]


def test_e2e_generate_judge_and_report(tmp_path):
    rag, gen = answering_rag(), FakeLLM(lambda s, u: {"answer": "x", "abstain": True})
    qs = [question("q1"), question("q2", answerable=False)]
    path = tmp_path / "rows.jsonl"
    e2e.generate_answers("rag", rag, gen, qs, path)
    e2e.generate_answers("cb", None, gen, qs, path)
    assert e2e.generate_answers("rag", rag, gen, qs, path) == []  # resumable: nothing left

    rows = e2e.load_rows(path)
    by = {(r["pipeline"], r["qid"]): r for r in rows}
    assert by[("rag", "q1")]["outcome"] is None  # needs the judge
    assert by[("rag", "q2")]["outcome"] == "hallucinated"
    assert by[("cb", "q2")]["outcome"] == "correct_abstention"
    assert by[("rag", "q1")]["context_recall"] == 1.0
    assert by[("rag", "q1")]["citation_precision"] == 1.0
    assert by[("rag", "q1")]["needs_faithfulness"] and not by[("cb", "q1")]["needs_faithfulness"]
    assert len(by[("rag", "q1")]["fragments"]) == 1  # only the cited passage

    judge = Judge(FakeLLM(judge_reply("correct", True)))
    assert e2e.judge_rows(rows, judge, path) == 0
    saved = {(r["pipeline"], r["qid"]): r for r in e2e.load_rows(path)}
    assert saved[("rag", "q1")]["outcome"] == "correct" and saved[("rag", "q1")]["faithful"]
    assert saved[("rag", "q2")]["faithful"] is True  # hallucinations are checked too

    report = e2e.write_report(
        rows, tmp_path / "r.md", title="t", generator=gen, baseline="cb", agreement="AGREEMENT"
    ).read_text(encoding="utf-8")
    assert "| `rag` |" in report and "/ 1k queries" in report
    assert "AGREEMENT" in report and "q2" in report and "Incomplete" not in report


def test_judge_stops_on_quota_and_resumes(tmp_path):
    rows = [
        {
            "pipeline": "p",
            "qid": f"q{i}",
            "outcome": None,
            "question": "q",
            "reference": "r",
            "answer": "a",
            "needs_faithfulness": False,
            "faithful": None,
            "type": "lookup",
            "answerable": True,
            "judge_reason": "",
            "context_recall": None,
            "latency_s": 1.0,
            "prompt_tokens": 1,
            "completion_tokens": 1,
        }
        for i in range(3)
    ]
    path = tmp_path / "rows.jsonl"
    calls = {"n": 0}

    def reply(system, user):
        calls["n"] += 1
        if calls["n"] == 2:
            raise QuotaExhausted("tokens per day")
        return {"reason": "", "verdict": "correct"}

    assert e2e.judge_rows(rows, Judge(FakeLLM(reply)), path) == 2
    saved = e2e.load_rows(path)
    assert [r["outcome"] for r in saved] == ["correct", None, None]
    report = e2e.write_report(
        saved, tmp_path / "r.md", title="t", generator=FakeLLM(reply), baseline="p"
    ).read_text(encoding="utf-8")
    assert "2 answers still to be judged" in report
    assert e2e.judge_rows(saved, Judge(FakeLLM(reply)), path) == 0


def test_question_generation_stops_on_quota():
    def reply(system, user):
        raise QuotaExhausted("requests per day")

    candidates, stats = generate(
        [DOC],
        CHUNKS,
        FakeLLM(reply),
        n_single=3,
        n_linked=0,
        n_unanswerable=0,
        used_sections=set(),
        min_words=3,
    )
    assert candidates == [] and "per day" in stats.stopped


def test_daily_quota_detection():
    assert _is_daily_quota(Exception("Rate limit reached on tokens per day (TPD)"))
    assert not _is_daily_quota(Exception("Rate limit reached on requests per minute"))


# --- Query transformation ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "reply", "n_variants"),
    [
        ("rewrite", {"query": "plazo de días hábiles"}, 1),
        ("hyde", {"passage": "El plazo será de quince días."}, 1),
        ("multi", {"queries": ["a", "b", "c", "d"]}, 4),  # original + 3
    ],
)
def test_query_transform(mode, reply, n_variants):
    bm25 = BM25()
    t = QueryTransform("t", base=bm25, llm=FakeLLM(lambda s, u: reply), mode=mode)
    t.index(CHUNKS)
    assert len(t.transform("¿plazo?")) == n_variants
    assert t.search(["¿plazo?"], 3)[0]


def test_query_transform_falls_back_on_bad_output():
    t = QueryTransform("t", base=BM25(), llm=FakeLLM(lambda s, u: None), mode="rewrite")
    assert t.transform("¿plazo?") == ["¿plazo?"]


# --- Question generation -----------------------------------------------------------------------


def test_find_quote_tolerates_whitespace_and_typography():
    a1 = SECTIONS["a1"]
    assert find_quote(a1, "El plazo es de quince días.") == "El plazo es de quince días."
    assert find_quote(a1, "  El plazo   es de quince días ") == "El plazo es de quince días"
    assert find_quote(a1, "El plazo es de treinta días.") is None
    assert find_quote(a1, "") is None


def test_lexical_overlap():
    assert lexical_overlap("¿Plazo de quince días?", "El plazo es de quince días.") == 1.0
    assert lexical_overlap("¿Cuánto tiempo tengo?", "El plazo es de quince días.") == 0.0


def test_generate_keeps_only_verified_quotes():
    def reply(system, user):
        if "NO puedan responderse" in user:
            return {"questions": [{"question": "¿IVA?", "reason": "otra ley"}]}
        if "Artículo 1" in user:
            return {
                "question": "¿Cuánto tengo?",
                "answer": "15 días",
                "type": "numeric",
                "quote": "El plazo es de quince días.",
            }
        if "Escala" in user:
            return {"question": "¿Tipo?", "answer": "9,5", "quote": "texto inventado"}
        return {"skip": True}

    candidates, stats = generate(
        [DOC],
        CHUNKS,
        FakeLLM(reply),
        n_single=10,
        n_linked=5,
        n_unanswerable=1,
        used_sections=set(),
        min_words=3,
    )
    kinds = [c["type"] for c in candidates]
    assert kinds.count("numeric") == 1 and kinds.count("unanswerable") == 1
    assert stats.misquoted["single"] == 1
    assert stats.skipped["single"] >= 1
    assert candidates[0]["evidence"][0]["ref"] == "LP#a1"


# --- Review and labeling -----------------------------------------------------------------------


def test_review_saves_decisions(tmp_path):
    path = tmp_path / "c.yaml"
    rows = [
        {
            "id": f"g{i}",
            "question": f"q{i}",
            "answer": "a",
            "type": "lookup",
            "evidence": [{"ref": "LP: Artículo 1", "quote": "El plazo es de quince días."}],
            "status": "pending",
        }
        for i in range(3)
    ] + [
        {
            "id": "g3",
            "question": "¿IVA?",
            "answer": "UNANSWERABLE",
            "type": "unanswerable",
            "status": "pending",
        }
    ]
    path.write_text(yaml.safe_dump(rows, allow_unicode=True), encoding="utf-8")
    answers = iter(["a", "x", "r", "e", "pregunta nueva", "", "s"])
    lines = []
    counts = review(path, [DOC], CHUNKS, ask=lambda _: next(answers), out=lines.append)
    saved = {r["id"]: r for r in yaml.safe_load(path.read_text(encoding="utf-8"))}
    assert saved["g0"]["status"] == "accepted"
    assert saved["g1"]["status"] == "rejected"
    assert saved["g2"]["question"] == "pregunta nueva" and saved["g2"]["status"] == "accepted"
    assert saved["g3"]["status"] == "pending"
    assert counts == {"accepted": 2, "rejected": 1, "pending": 1}
    assert any("BM25 top 5" in line for line in lines)


def test_kappa():
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0


def test_label_and_agreement(tmp_path):
    rows = [
        {
            "pipeline": "p",
            "qid": f"q{i}",
            "outcome": o,
            "question": "q",
            "reference": "r",
            "answer": "a",
        }
        for i, o in enumerate(["correct", "incorrect", "partial", "hallucinated"])
    ]
    path = tmp_path / "labels.jsonl"
    answers = iter(["c", "c", "q"])
    assert label(rows, path, n=10, ask=lambda _: next(answers), out=lambda _: None) == 2
    labels = load_labels(path)
    assert len(labels) == 2
    text = agreement(rows, labels)
    assert text.startswith("2 answers labelled by hand")
    assert agreement(rows, {}) is None


# --- Provider errors -----------------------------------------------------------------------------


def _client_raising(exc):
    from types import SimpleNamespace

    def create(**kwargs):
        raise exc

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _api_error(cls, status: int, message: str):
    import httpx

    response = httpx.Response(status, request=httpx.Request("POST", "http://test"))
    return cls(message, response=response, body=None)


def test_invalid_json_from_provider_is_unparseable_output(monkeypatch):
    import openai

    from rag_ablation_bench.llm import LLM

    llm = LLM(ModelConfig("m", "p", "m", "http://test", None), cache_dir=None)
    llm._client = _client_raising(
        _api_error(openai.BadRequestError, 400, "Failed to validate JSON")
    )
    with pytest.raises(ValueError):
        llm.ask_json("s", "u")


def test_daily_quota_is_not_retried(monkeypatch):
    import openai

    from rag_ablation_bench import llm as llm_module

    slept = []
    monkeypatch.setattr(llm_module.time, "sleep", slept.append)
    llm = llm_module.LLM(ModelConfig("m", "p", "m", "http://test", None), cache_dir=None)
    llm._client = _client_raising(
        _api_error(openai.RateLimitError, 429, "Rate limit reached on tokens per day (TPD)")
    )
    with pytest.raises(QuotaExhausted):
        llm.ask_json("s", "u")
    assert slept == []  # gave up at once instead of backing off
