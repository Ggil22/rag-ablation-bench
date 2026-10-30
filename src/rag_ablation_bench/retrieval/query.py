"""Query transformation with an LLM before retrieval.

Users ask "¿cuántos días libres me dan si me caso?"; the law says "permiso retribuido...
en caso de matrimonio". These wrappers bridge that vocabulary gap in three ways:

- rewrite: one reformulation in the terminology of the law.
- hyde:    a hypothetical passage of the law that would answer the question (Gao et al.,
           2022); dense retrievers then match passage against passage.
- multi:   several reformulations, retrieved separately and fused with RRF, together with
           the original question.

The transformation happens inside `search`, so a reranker stacked on top still scores
chunks against the user's original question.
"""

from __future__ import annotations

from rag_ablation_bench.llm import LLM
from rag_ablation_bench.retrieval.base import Ranking, Retriever
from rag_ablation_bench.retrieval.fusion import rrf
from rag_ablation_bench.schema import Chunk

SYSTEM = (
    "Eres un experto en legislación española. Ayudas a un buscador de normas a encontrar "
    "los artículos que responden a la pregunta de un ciudadano. Responde solo con JSON."
)
PROMPTS = {
    "rewrite": (
        "Reformula esta pregunta como una consulta de búsqueda que use la terminología "
        "jurídica con la que la ley regularía el asunto (por ejemplo, 'permiso retribuido' en "
        "lugar de 'días libres'). No inventes números de artículo ni cifras.\n\n"
        'Pregunta: {question}\n\nDevuelve {{"query": "..."}}'
    ),
    "hyde": (
        "Escribe un fragmento breve (2-4 frases) de un artículo de una ley española que "
        "respondería a esta pregunta, con el estilo y vocabulario de un texto legal. Las cifras "
        "pueden ser inexactas; lo importante es el vocabulario.\n\n"
        'Pregunta: {question}\n\nDevuelve {{"passage": "..."}}'
    ),
    "multi": (
        "Escribe {n} consultas de búsqueda distintas para encontrar en la legislación española "
        "la respuesta a esta pregunta: usa terminología jurídica y enfoques diferentes "
        "(la institución jurídica, el derecho u obligación, el plazo o cuantía). No inventes "
        "números de artículo.\n\n"
        'Pregunta: {question}\n\nDevuelve {{"queries": ["...", "..."]}}'
    ),
}


class QueryTransform:
    def __init__(
        self,
        name: str,
        *,
        base: Retriever,
        llm: LLM,
        mode: str,
        n_queries: int = 3,
        depth: int = 100,
    ):
        if mode not in PROMPTS:
            raise ValueError(f"unknown query transform mode {mode!r}")
        self.name, self.base, self.llm, self.mode = name, base, llm, mode
        self.n_queries, self.depth = n_queries, depth
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    def transform(self, question: str) -> list[str]:
        prompt = PROMPTS[self.mode].format(question=question, n=self.n_queries)
        try:
            data, response = self.llm.ask_json(SYSTEM, prompt)
        except ValueError:  # unparseable answer: fall back to the original question
            return [question]
        self.usage["prompt_tokens"] += response.prompt_tokens
        self.usage["completion_tokens"] += response.completion_tokens
        self.usage["calls"] += 1
        if self.mode == "rewrite":
            return [str(data.get("query") or question)]
        if self.mode == "hyde":
            # Question and passage together: the passage adds vocabulary, the question
            # keeps the specifics the passage may have got wrong.
            return [f"{question}\n{data.get('passage', '')}"]
        queries = [str(q) for q in data.get("queries", []) if str(q).strip()]
        return [question, *queries[: self.n_queries]]

    def index(self, chunks: list[Chunk]) -> None:
        self.base.index(chunks)

    def search(self, queries: list[str], k: int) -> list[Ranking]:
        results = []
        for question in queries:
            variants = self.transform(question)
            if len(variants) == 1:
                results.append(self.base.search(variants, k)[0])
            else:
                rankings = self.base.search(variants, self.depth)
                results.append(rrf(rankings)[:k])
        return results
