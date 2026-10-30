import json
from contextlib import contextmanager

import pytest

fastapi = pytest.importorskip("fastapi")  # the `serve` extra
from fastapi.testclient import TestClient  # noqa: E402

from rag_ablation_bench.rag import RAG  # noqa: E402
from rag_ablation_bench.retrieval import BM25  # noqa: E402
from rag_ablation_bench.serve.app import app  # noqa: E402
from rag_ablation_bench.serve.observability import Tracer  # noqa: E402
from tests.test_llm_flows import CHUNKS, DOC, FakeLLM  # noqa: E402


class FakeLangfuse:
    def __init__(self):
        self.events = []

    @contextmanager
    def _cm(self, name, kwargs):
        self.events.append(("start", name, kwargs.get("as_type")))
        span = self

        yield span
        self.events.append(("end", name))

    def start_as_current_observation(self, *, name, **kwargs):
        return self._cm(name, kwargs)

    def update(self, **kwargs):
        self.events.append(("update", sorted(kwargs)))

    def flush(self):
        pass


@pytest.fixture
def client(tmp_path):
    bm25 = BM25()
    bm25.index(CHUNKS)

    def reply(system, user):
        if "inventa" in user:
            return None  # unparseable generator output
        return {"answer": "Quince días (art. 1).", "citations": [1], "abstain": False}

    app.state.rag = RAG(bm25, CHUNKS, FakeLLM(reply), budget=300)
    app.state.urls = {DOC.id: DOC.url}
    app.state.pipeline = "test"
    app.state.tracer = Tracer(tmp_path / "requests.jsonl", FakeLangfuse())
    return TestClient(app)  # no `with`: skips the lifespan that loads the real pipeline


def test_health(client):
    assert client.get("/health").json() == {
        "status": "ok",
        "pipeline": "test",
        "chunks": len(CHUNKS),
    }


def test_ask_returns_answer_citations_and_costs(client, tmp_path):
    r = client.post("/ask", json={"question": "¿Cuál es el plazo en días?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Quince días (art. 1)." and not body["abstain"]
    assert body["citations"][0]["url"] == DOC.url
    assert body["prompt_tokens"] == 100 and body["cost_usd"] == pytest.approx(120 / 1e6)
    assert body["total_ms"] >= body["retrieval_ms"]

    [logged] = [json.loads(line) for line in (tmp_path / "requests.jsonl").open(encoding="utf-8")]
    assert logged["request_id"] == body["request_id"] and logged["cited"]

    events = app.state.tracer.langfuse.events
    starts = [(e[1], e[2]) for e in events if e[0] == "start"]
    assert starts == [("ask", "span"), ("retrieval", "retriever"), ("generation", "generation")]
    assert events[-1] == ("end", "ask")  # children closed inside the root


def test_generator_failure_is_502(client):
    r = client.post("/ask", json={"question": "inventa algo"})
    assert r.status_code == 502


def test_validation(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_tracer_without_langfuse_or_log(tmp_path):
    tracer = Tracer(None)
    with tracer.observe("x") as span:
        span.update(output="ignored")
    tracer.log_request({"a": 1})  # no log path: nothing happens
    tracer.flush()
