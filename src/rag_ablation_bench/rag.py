"""The RAG pipeline: retrieve, build the context, generate a cited answer or abstain."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rag_ablation_bench.evaluation.metrics import within_budget
from rag_ablation_bench.llm import LLM
from rag_ablation_bench.retrieval import Retriever
from rag_ablation_bench.schema import Chunk

SYSTEM = (
    "Eres un asistente que responde preguntas sobre legislación española usando ÚNICAMENTE "
    "los fragmentos de normas que se te proporcionan. Reglas:\n"
    "- Responde en español, de forma breve y precisa, citando la norma y el artículo.\n"
    "- Usa solo la información de los fragmentos; no uses conocimiento propio.\n"
    "- Si los fragmentos no contienen la respuesta, abstente: no adivines ni completes con "
    "lo que sepas.\n"
    "- Indica en 'citations' los números de los fragmentos que usas.\n"
    "Responde solo con JSON."
)
USER = """Fragmentos:

{context}

Pregunta: {question}

Devuelve {{"answer": "...", "citations": [1, 2], "abstain": false}}
Si abstienes, devuelve {{"answer": "", "citations": [], "abstain": true}}"""


@dataclass
class Answer:
    question: str
    answer: str
    abstain: bool
    citations: list[int]  # indices into `context`
    context: list[Chunk]
    retrieval_ms: float
    generation_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None
    extra: dict = field(default_factory=dict)


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{i}] {c.header}\n{c.text}" for i, c in enumerate(chunks, 1))


class RAG:
    def __init__(
        self,
        retriever: Retriever,
        chunks: list[Chunk],
        llm: LLM,
        *,
        budget: int = 2000,
        depth: int = 20,
    ):
        self.retriever, self.chunks, self.llm = retriever, chunks, llm
        self.budget, self.depth = budget, depth

    def retrieve(self, question: str) -> tuple[list[Chunk], float]:
        t0 = time.perf_counter()
        ranking = self.retriever.search([question], self.depth)[0]
        ms = 1000 * (time.perf_counter() - t0)
        return within_budget([self.chunks[i] for i, _ in ranking], self.budget), ms

    def answer(self, question: str) -> Answer:
        context, retrieval_ms = self.retrieve(question)
        return self.generate(question, context, retrieval_ms)

    def generate(self, question: str, context: list[Chunk], retrieval_ms: float = 0.0) -> Answer:
        prompt = USER.format(context=format_context(context), question=question)
        try:
            data, response = self.llm.ask_json(SYSTEM, prompt)
        except ValueError as exc:  # unparseable output counts as a failed answer
            return Answer(question, "", False, [], context, retrieval_ms, 0.0, error=str(exc))
        citations = [
            int(c) - 1
            for c in data.get("citations") or []
            if str(c).isdigit() and 1 <= int(c) <= len(context)
        ]
        return Answer(
            question=question,
            answer=str(data.get("answer") or ""),
            abstain=bool(data.get("abstain")),
            citations=citations,
            context=context,
            retrieval_ms=retrieval_ms,
            generation_s=response.latency_s,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
