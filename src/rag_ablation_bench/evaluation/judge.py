"""Scoring generated answers.

Outcomes:
- unanswerable questions are scored without an LLM: abstaining is `correct_abstention`,
  answering anything is `hallucinated`;
- an abstention on an answerable question is `false_abstention`;
- otherwise an LLM judge compares the answer with the reference: `correct`, `partial` or
  `incorrect`.

Separately, the judge checks *faithfulness*: whether every claim in the answer is supported
by the passages the answer cites (by the whole retrieved context when it cites none). This
is stricter than checking against everything retrieved, and much cheaper in tokens. An answer can be correct but unfaithful (right from the model's
own memory, which a legal assistant should not rely on), or faithful but incorrect (the
context retrieved was the wrong provision).

The judge is an LLM and can be wrong, so its verdicts are validated against human labels
(`rab label`, `rab agreement`) before any conclusion rests on them.
"""

from __future__ import annotations

from rag_ablation_bench.llm import LLM

OUTCOMES = (
    "correct",
    "partial",
    "incorrect",
    "false_abstention",
    "correct_abstention",
    "hallucinated",
    "error",
)
# Outcome -> score used for the headline accuracy.
SCORE = {"correct": 1.0, "partial": 0.5, "correct_abstention": 1.0}

SYSTEM = (
    "Eres un evaluador riguroso de respuestas sobre legislación española. Juzgas con "
    "criterios jurídicos, no de estilo. Responde solo con JSON."
)
CORRECTNESS = """Pregunta: {question}

Respuesta de referencia (correcta): {reference}

Respuesta a evaluar: {answer}

Compara la respuesta a evaluar con la referencia en cuanto a contenido jurídico:
- "correct": contiene la información esencial de la referencia (cifras, plazos, condiciones) y no afirma nada que la contradiga. Puede añadir detalles correctos o decirlo con otras palabras.
- "partial": acierta en parte pero omite una condición o dato esencial, o es imprecisa.
- "incorrect": da una cifra, plazo o regla distinta, o no responde a lo preguntado.

Devuelve {{"reason": "...", "verdict": "correct|partial|incorrect"}}"""

FAITHFULNESS = """Fragmentos de normas que cita el asistente:

{context}

Respuesta del asistente: {answer}

¿Está cada afirmación de la respuesta respaldada por los fragmentos? Ignora la forma; fíjate en cifras, plazos, condiciones y referencias a artículos. Una afirmación que no aparece en los fragmentos no está respaldada aunque sea cierta.

Devuelve {{"reason": "...", "supported": true|false}}"""


def deterministic_outcome(
    *, answerable: bool, abstain: bool, answer: str, error: str | None
) -> str | None:
    """The outcome when no judge is needed, else None."""
    if error:
        return "error"
    if not answerable:
        return "correct_abstention" if abstain else "hallucinated"
    if abstain or not answer.strip():
        return "false_abstention"
    return None


class Judge:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def _ask(self, prompt: str) -> dict:
        data, response = self.llm.ask_json(SYSTEM, prompt)
        self.usage["prompt_tokens"] += response.prompt_tokens
        self.usage["completion_tokens"] += response.completion_tokens
        return data

    def correctness(self, question: str, reference: str, answer: str) -> tuple[str, str]:
        """(verdict, reason); verdict is correct, partial, incorrect, or error."""
        try:
            data = self._ask(
                CORRECTNESS.format(question=question, reference=reference, answer=answer)
            )
        except ValueError as exc:
            return "error", f"judge output: {exc}"
        verdict = str(data.get("verdict", "")).strip().lower()
        if verdict not in ("correct", "partial", "incorrect"):
            return "error", f"judge verdict {verdict!r}"
        return verdict, str(data.get("reason", ""))

    def faithfulness(self, answer: str, fragments: list[str]) -> bool | None:
        """Whether the passages support every claim; None if the judge output is unusable."""
        context = "\n\n".join(f"[{i}] {f}" for i, f in enumerate(fragments, 1))
        try:
            data = self._ask(FAITHFULNESS.format(context=context, answer=answer))
        except ValueError:
            return None
        supported = data.get("supported")
        return supported if isinstance(supported, bool) else None
