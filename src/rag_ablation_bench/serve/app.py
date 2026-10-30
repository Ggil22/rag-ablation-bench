"""HTTP API serving one pipeline from configs/pipelines.yaml.

    uv run --extra serve uvicorn rag_ablation_bench.serve.app:app --port 8080

Environment:
    RAB_PIPELINE   pipeline name (default: the `serve` entry of pipelines.yaml)
    RAB_DATA       data directory with the processed corpus (default: data)
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_ablation_bench.chunking import chunk_documents
from rag_ablation_bench.llm import LLM, load_model_configs
from rag_ablation_bench.rag import RAG
from rag_ablation_bench.retrieval import RetrieverFactory
from rag_ablation_bench.schema import load_documents
from rag_ablation_bench.serve.observability import Tracer

CONFIGS = Path(os.environ.get("RAB_CONFIGS", "configs"))


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class Citation(BaseModel):
    section_id: str
    header: str
    url: str
    text: str


class AskResponse(BaseModel):
    request_id: str
    answer: str
    abstain: bool
    citations: list[Citation]
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))


def build_rag(pipeline: str, data_dir: Path) -> tuple[RAG, dict]:
    from dotenv import load_dotenv

    load_dotenv(Path(".env"))
    config = _load("pipelines")
    spec = config["pipelines"][pipeline]
    if spec.get("closed_book"):
        raise ValueError("the closed-book baseline cannot be served")
    docs = load_documents(data_dir / "processed" / "boe" / "documents.jsonl")
    params = dict(_load("chunking")[spec["chunking"]])
    chunks = chunk_documents(docs, params.pop("strategy"), **params)
    models = load_model_configs(CONFIGS / "models.yaml")
    factory = RetrieverFactory(_load("retrieval"), llm=lambda m: LLM(models[m]))
    retriever = factory(spec["retriever"])
    retriever.index(chunks)
    retriever.search(["calentamiento"], 1)  # load models now, not on the first request
    rag = RAG(retriever, chunks, LLM(models[config["generator"]]), budget=config["budget"])
    urls = {d.id: d.url for d in docs}
    return rag, urls


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = _load("pipelines")
    pipeline = os.environ.get("RAB_PIPELINE", config.get("serve", config["baseline"]))
    rag, urls = build_rag(pipeline, Path(os.environ.get("RAB_DATA", "data")))
    app.state.rag, app.state.urls, app.state.pipeline = rag, urls, pipeline
    app.state.tracer = Tracer.from_env()
    yield
    app.state.tracer.flush()


app = FastAPI(title="rag-ablation-bench", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "pipeline": app.state.pipeline, "chunks": len(app.state.rag.chunks)}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    rag: RAG = app.state.rag
    tracer: Tracer = app.state.tracer
    request_id = uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    with tracer.observe(
        "ask",
        input=request.question,
        metadata={"pipeline": app.state.pipeline, "request_id": request_id},
    ) as root:
        with tracer.observe("retrieval", as_type="retriever", input=request.question) as span:
            context, retrieval_ms = rag.retrieve(request.question)
            span.update(output=[{"chunk": c.id, "header": c.header} for c in context])
        with tracer.observe("generation", as_type="generation", model=rag.llm.cfg.model) as gen:
            answer = rag.generate(request.question, context, retrieval_ms)
            gen.update(
                output=answer.error or answer.answer,
                usage_details={"input": answer.prompt_tokens, "output": answer.completion_tokens},
                level="ERROR" if answer.error else None,
            )
        root.update(output={"answer": answer.answer, "abstain": answer.abstain})
    total_ms = 1000 * (time.perf_counter() - t0)
    if answer.error:
        raise HTTPException(status_code=502, detail=f"generator failed: {answer.error}")
    cited = [answer.context[i] for i in answer.citations]
    response = AskResponse(
        request_id=request_id,
        answer=answer.answer,
        abstain=answer.abstain,
        citations=[
            Citation(
                section_id=c.section_ids[0],
                header=c.header,
                url=app.state.urls[c.doc_id],
                text=c.text,
            )
            for c in cited
        ],
        retrieval_ms=answer.retrieval_ms,
        generation_ms=1000 * answer.generation_s,
        total_ms=total_ms,
        prompt_tokens=answer.prompt_tokens,
        completion_tokens=answer.completion_tokens,
        cost_usd=rag.llm.cfg.cost(answer.prompt_tokens, answer.completion_tokens),
    )
    body = response.model_dump()
    tracer.log_request(
        {
            "pipeline": app.state.pipeline,
            "question": request.question,
            "context": [c.id for c in context],
            **{k: v for k, v in body.items() if k != "citations"},
            "cited": [c["section_id"] for c in body["citations"]],
        }
    )
    return response
