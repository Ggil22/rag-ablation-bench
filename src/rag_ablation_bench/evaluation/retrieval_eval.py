"""Run retrieval experiments over chunking x retriever and report them."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from rag_ablation_bench.evaluation.dataset import Question
from rag_ablation_bench.evaluation.metrics import KS, is_relevant, score_ranking
from rag_ablation_bench.evaluation.stats import bootstrap_ci, paired_diff_ci
from rag_ablation_bench.retrieval import Retriever
from rag_ablation_bench.schema import Chunk

DEPTH = max(KS)
HEADLINE = ("recall@5", "recall@1000w", "complete@5", "mrr@10")


def run(
    chunk_sets: dict[str, list[Chunk]],
    retrievers: list[str],
    make: Callable[[str], Retriever],
    questions: list[Question],
    out_path: Path,
) -> list[dict]:
    """Evaluate every (chunking, retriever) pair on the answerable questions."""
    answerable = [q for q in questions if q.answerable]
    texts = [q.question for q in answerable]
    rows = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for chunking, chunks in chunk_sets.items():
            for name in retrievers:
                retriever = make(name)
                t0 = time.perf_counter()
                retriever.index(chunks)
                index_s = time.perf_counter() - t0
                retriever.search(texts[:1], 1)  # warm-up: lazy model loading, CUDA init
                t0 = time.perf_counter()
                rankings = retriever.search(texts, DEPTH)
                ms_per_query = 1000 * (time.perf_counter() - t0) / len(texts)
                for q, ranking in zip(answerable, rankings, strict=True):
                    ranked = [chunks[i] for i, _ in ranking]
                    row = {
                        "chunking": chunking,
                        "retriever": name,
                        "qid": q.id,
                        "type": q.type,
                        **score_ranking(ranked, q.facts),
                        "top": [[c.id, is_relevant(c, q.facts)] for c in ranked[:10]],
                        "ms_per_query": ms_per_query,
                        "index_s": index_s,
                    }
                    rows.append(row)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(
                    f"{chunking:<15} {name:<22} recall@5 "
                    f"{_mean(rows, chunking, name, 'recall@5'):.3f}  {ms_per_query:7.1f} ms/q"
                )
    return rows


def _mean(rows: list[dict], chunking: str, retriever: str, metric: str) -> float:
    vals = [r[metric] for r in rows if r["chunking"] == chunking and r["retriever"] == retriever]
    return sum(vals) / len(vals)


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _fmt(mean: float, lo: float, hi: float) -> str:
    return f"{mean:.2f} [{lo:.2f}, {hi:.2f}]"


def write_report(
    rows: list[dict],
    out_path: Path,
    *,
    title: str,
    baseline: tuple[str, str],
    llm_retrievers: frozenset[str] = frozenset(),
):
    """Markdown report: headline metrics with 95% bootstrap CIs, paired differences against a
    baseline configuration, and recall by question type."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["chunking"], r["retriever"])].append(r)
    for g in groups.values():
        g.sort(key=lambda r: r["qid"])
    n_questions = len(next(iter(groups.values())))

    lines = [
        f"# {title}",
        "",
        f"{n_questions} answerable questions. Mean with 95% bootstrap interval.",
        "`recall@k`: share of the facts the answer needs found in the top k chunks.",
        "`recall@1000w`: same, with the top chunks that fit in 1,000 words of context",
        "(the fair comparison across chunk sizes). `complete@5`: all facts in the top 5.",
        "† Retrievers that call an LLM: their wall time is dominated by free-tier throttling",
        "or by the response cache, so it is not reported; add one LLM call to the latency.",
        "",
        "| Chunking | Retriever | " + " | ".join(HEADLINE) + " | ms/query |",
        "|---|---|" + "---|" * (len(HEADLINE) + 1),
    ]
    order = sorted(groups, key=lambda key: -_avg(groups[key], "recall@1000w"))
    for key in order:
        g = groups[key]
        cells = [_fmt(*bootstrap_ci([r[m] for r in g])) for m in HEADLINE]
        lines.append(
            f"| `{key[0]}` | `{key[1]}` | "
            + " | ".join(cells)
            + (" | LLM† |" if key[1] in llm_retrievers else f" | {g[0]['ms_per_query']:.0f} |")
        )

    if baseline in groups:
        base = groups[baseline]
        lines += [
            "",
            f"## Difference against `{baseline[0]}` + `{baseline[1]}`",
            "",
            "Paired over the same questions; **bold** when the 95% interval excludes zero.",
            "",
            "| Chunking | Retriever | Δ recall@1000w | Δ mrr@10 |",
            "|---|---|---|---|",
        ]
        for key in order:
            if key == baseline:
                continue
            cells = []
            for m in ("recall@1000w", "mrr@10"):
                mean, lo, hi = paired_diff_ci([r[m] for r in groups[key]], [r[m] for r in base])
                cell = f"{mean:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                cells.append(f"**{cell}**" if lo > 0 or hi < 0 else cell)
            lines.append(f"| `{key[0]}` | `{key[1]}` | " + " | ".join(cells) + " |")

    types = sorted({r["type"] for r in rows})
    lines += [
        "",
        "## recall@1000w by question type",
        "",
        "| Chunking | Retriever | "
        + " | ".join(f"{t} (n={_count(rows, t, order[0])})" for t in types)
        + " |",
        "|---|---|" + "---|" * len(types),
    ]
    for key in order:
        cells = [
            f"{_avg([r for r in groups[key] if r['type'] == t], 'recall@1000w'):.2f}" for t in types
        ]
        lines.append(f"| `{key[0]}` | `{key[1]}` | " + " | ".join(cells) + " |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _avg(rows: list[dict], metric: str) -> float:
    return sum(r[metric] for r in rows) / len(rows) if rows else float("nan")


def _count(rows: list[dict], qtype: str, key: tuple[str, str]) -> int:
    return sum(1 for r in rows if r["type"] == qtype and (r["chunking"], r["retriever"]) == key)
