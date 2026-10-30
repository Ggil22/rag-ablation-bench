"""End-to-end evaluation: retrieval + generation + judge, per pipeline."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from rag_ablation_bench.evaluation.dataset import Question
from rag_ablation_bench.evaluation.judge import OUTCOMES, SCORE, Judge, deterministic_outcome
from rag_ablation_bench.evaluation.metrics import facts_found, is_relevant
from rag_ablation_bench.evaluation.stats import bootstrap_ci, paired_diff_ci
from rag_ablation_bench.llm import LLM, QuotaExhausted
from rag_ablation_bench.rag import RAG, Answer

CLOSED_BOOK_SYSTEM = (
    "Eres un asistente que responde preguntas sobre legislación española vigente con tu "
    "propio conocimiento. Responde en español, de forma breve y precisa, citando la norma y "
    "el artículo. Si no estás seguro de la respuesta, abstente. Responde solo con JSON."
)
CLOSED_BOOK_USER = """Pregunta: {question}

Devuelve {{"answer": "...", "citations": [], "abstain": false}}
Si abstienes, devuelve {{"answer": "", "citations": [], "abstain": true}}"""


def closed_book(llm: LLM, question: str) -> Answer:
    try:
        data, r = llm.ask_json(CLOSED_BOOK_SYSTEM, CLOSED_BOOK_USER.format(question=question))
    except ValueError as exc:
        return Answer(question, "", False, [], [], 0.0, 0.0, error=str(exc))
    return Answer(
        question,
        str(data.get("answer") or ""),
        bool(data.get("abstain")),
        [],
        [],
        0.0,
        r.latency_s,
        r.prompt_tokens,
        r.completion_tokens,
    )


def generate_answers(
    name: str,
    rag: RAG | None,
    generator: LLM,
    questions: list[Question],
    path: Path,
) -> list[dict]:
    """Answer every question not yet answered by this pipeline, appending rows to `path`.
    `rag=None` runs the closed-book baseline. Outcomes that need no judge are set here; the
    rest are left as None for `judge_rows`. Raises QuotaExhausted with all prior rows saved.
    """
    done = {r["qid"] for r in load_rows(path) if r["pipeline"] == name} if path.exists() else set()
    todo = [q for q in questions if q.id not in done]
    rows = []
    with path.open("a", encoding="utf-8") as out:
        for n, q in enumerate(todo, 1):
            a = rag.answer(q.question) if rag else closed_book(generator, q.question)
            cited = [a.context[i] for i in a.citations]
            outcome = deterministic_outcome(
                answerable=q.answerable, abstain=a.abstain, answer=a.answer, error=a.error
            )
            # Passages the faithfulness check reads: the cited ones, or all if none cited.
            fragments = [f"{c.header}\n{c.text}" for c in (cited or a.context)]
            row = {
                "pipeline": name,
                "qid": q.id,
                "type": q.type,
                "answerable": q.answerable,
                "outcome": outcome,
                "judge_reason": a.error or "",
                "faithful": None,
                # Any substantive RAG answer, including a hallucinated one: was it at least
                # grounded in what it cites?
                "needs_faithfulness": bool(rag)
                and not a.error
                and not a.abstain
                and bool(a.answer.strip()),
                "question": q.question,
                "reference": q.answer,
                "answer": a.answer,
                "abstain": a.abstain,
                "context": [c.id for c in a.context],
                "cited": [c.id for c in cited],
                "fragments": fragments if rag else [],
                # Of the chunks the answer cites, how many hold gold evidence.
                "citation_precision": (
                    sum(is_relevant(c, q.facts) for c in cited) / len(cited)
                    if cited and q.answerable
                    else None
                ),
                "context_recall": (
                    facts_found(q.facts, a.context) / len(q.facts) if q.answerable and rag else None
                ),
                "retrieval_ms": a.retrieval_ms,
                "generation_s": a.generation_s,
                "latency_s": a.retrieval_ms / 1000 + a.generation_s,
                "prompt_tokens": a.prompt_tokens,
                "completion_tokens": a.completion_tokens,
            }
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"  {name:<24} {n:>3}/{len(todo)} {q.id:<10} {outcome or 'to judge'}", flush=True)
    return rows


def judge_rows(rows: list[dict], judge: Judge, path: Path) -> int:
    """Fill in pending verdicts, saving after every row so a run cut short by a provider
    quota resumes where it stopped. Returns how many rows are still pending."""
    try:
        for r in rows:
            if r["outcome"] is None:
                r["outcome"], r["judge_reason"] = judge.correctness(
                    r["question"], r["reference"], r["answer"]
                )
                save_rows(rows, path)
            if r.get("needs_faithfulness") and r["faithful"] is None:
                r["faithful"] = judge.faithfulness(r["answer"], r["fragments"])
                r["needs_faithfulness"] = False
                save_rows(rows, path)
    except QuotaExhausted as exc:
        print(f"Judge stopped: {exc}")
    return sum(1 for r in rows if r["outcome"] is None or r.get("needs_faithfulness"))


def save_rows(rows: list[dict], path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    lines = (json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    tmp.write_text("".join(lines), encoding="utf-8")
    tmp.replace(path)


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _pct(xs: list[float]) -> str:
    return f"{100 * sum(xs) / len(xs):.0f}%" if xs else "–"


def write_report(
    rows: list[dict],
    path: Path,
    *,
    title: str,
    generator: LLM,
    baseline: str,
    agreement: str | None = None,
) -> Path:
    pending = [r for r in rows if r["outcome"] is None]
    rows = [{**r, "score": SCORE.get(r["outcome"], 0.0)} for r in rows if r["outcome"] is not None]
    by_pipe: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_pipe[r["pipeline"]].append(r)
    for g in by_pipe.values():
        g.sort(key=lambda r: r["qid"])
    order = sorted(by_pipe, key=lambda p: -np.mean([r["score"] for r in by_pipe[p]]))
    n_ans = sum(r["answerable"] for r in by_pipe[order[0]])
    n_un = len(by_pipe[order[0]]) - n_ans

    lines = [
        f"# {title}",
        "",
        *(
            [f"> **Incomplete:** {len(pending)} answers still to be judged (`rab judge`).", ""]
            if pending
            else []
        ),
        f"Generator `{generator.cfg.model}`. {n_ans} answerable and {n_un} unanswerable "
        "questions. Accuracy: correct = 1, partial = 0.5, correct abstention = 1; mean with "
        "95% bootstrap interval.",
        "",
        "| Pipeline | n | Accuracy | Correct (answerable) | Hallucinated (unanswerable) | "
        "False abstention | Faithful | Context recall | p50 / p95 latency | Tokens / query |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for p in order:
        g = by_pipe[p]
        ans = [r for r in g if r["answerable"]]
        un = [r for r in g if not r["answerable"]]
        faithful = [r["faithful"] for r in g if r["faithful"] is not None]
        recall = [r["context_recall"] for r in ans if r["context_recall"] is not None]
        lat = np.array([r["latency_s"] for r in g])
        tokens = np.mean([r["prompt_tokens"] + r["completion_tokens"] for r in g])
        acc = bootstrap_ci([r["score"] for r in g])
        cost = generator.cfg.cost(
            int(np.mean([r["prompt_tokens"] for r in g])),
            int(np.mean([r["completion_tokens"] for r in g])),
        )
        cost_txt = f" (${cost * 1000:.2f} / 1k queries)" if cost is not None else ""
        lines.append(
            f"| `{p}` | {len(g)} | {acc[0]:.2f} [{acc[1]:.2f}, {acc[2]:.2f}] "
            f"| {_pct([r['outcome'] == 'correct' for r in ans])} "
            f"| {_pct([r['outcome'] == 'hallucinated' for r in un])} "
            f"| {_pct([r['outcome'] == 'false_abstention' for r in ans])} "
            f"| {_pct(faithful)} | {_pct(recall)} "
            f"| {np.percentile(lat, 50):.1f} s / {np.percentile(lat, 95):.1f} s "
            f"| {tokens:,.0f}{cost_txt} |"
        )

    if baseline in by_pipe:
        lines += [
            "",
            f"## Accuracy difference against `{baseline}`",
            "",
            "Paired over the same questions; **bold** when the 95% interval excludes zero.",
            "",
            "| Pipeline | Δ accuracy | Questions |",
            "|---|---|---|",
        ]
        base = {r["qid"]: r["score"] for r in by_pipe[baseline]}
        for p in order:
            if p == baseline:
                continue
            # Only questions both pipelines have answered (and the judge has scored).
            mine = {r["qid"]: r["score"] for r in by_pipe[p]}
            common = sorted(mine.keys() & base.keys())
            mean, lo, hi = paired_diff_ci([mine[q] for q in common], [base[q] for q in common])
            cell = f"{mean:+.2f} [{lo:+.2f}, {hi:+.2f}]"
            bold = f"**{cell}**" if lo > 0 or hi < 0 else cell
            lines.append(f"| `{p}` | {bold} | {len(common)} |")

    lines += [
        "",
        "## Outcomes",
        "",
        "| Pipeline | " + " | ".join(OUTCOMES) + " |",
        "|---|" + "---|" * len(OUTCOMES),
    ]
    for p in order:
        c = Counter(r["outcome"] for r in by_pipe[p])
        lines.append(f"| `{p}` | " + " | ".join(str(c.get(o, 0)) for o in OUTCOMES) + " |")

    types = sorted({r["type"] for r in rows})
    lines += [
        "",
        "## Accuracy by question type",
        "",
        "| Pipeline | " + " | ".join(types) + " |",
        "|---|" + "---|" * len(types),
    ]
    for p in order:
        cells = [
            f"{np.mean([r['score'] for r in by_pipe[p] if r['type'] == t]):.2f}" for t in types
        ]
        lines.append(f"| `{p}` | " + " | ".join(cells) + " |")

    if agreement:
        lines += ["", "## Judge validation", "", agreement]

    best = order[0]
    failures = [r for r in by_pipe[best] if r["score"] < 1]
    lines += ["", f"## Failures of the best pipeline (`{best}`)", ""]
    for r in failures:
        lines += [
            f"**{r['qid']}** ({r['type']}, {r['outcome']}): {r['question']}",
            f"- Reference: {r['reference']}",
            f"- Answer: {r['answer'] or '(abstained)'}",
        ]
        if r["judge_reason"]:
            lines.append(f"- Judge: {r['judge_reason']}")
        if r["context_recall"] is not None:
            lines.append(f"- Evidence in context: {r['context_recall']:.0%}")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
