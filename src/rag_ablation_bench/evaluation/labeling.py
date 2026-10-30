"""Human labels for a sample of answers, and agreement between the LLM judge and them.

The labeler sees question, reference and answer, but not the judge's verdict.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Callable
from pathlib import Path

LABELS = {"c": "correct", "p": "partial", "i": "incorrect"}
JUDGED = set(LABELS.values())


def load_labels(path: Path) -> dict[tuple[str, str], str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {(r["pipeline"], r["qid"]): r["label"] for r in rows}


def label(
    rows: list[dict],
    path: Path,
    *,
    n: int,
    seed: int = 0,
    ask: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
) -> int:
    """Label up to `n` judged answers, sampled at random across pipelines."""
    done = load_labels(path)
    pool = [r for r in rows if r["outcome"] in JUDGED and (r["pipeline"], r["qid"]) not in done]
    random.Random(seed).shuffle(pool)
    todo = pool[: max(0, n - len(done))]
    out(f"{len(done)} labelled, {len(todo)} to go. [c]orrect [p]artial [i]ncorrect [q]uit\n")
    count = 0
    with path.open("a", encoding="utf-8") as f:
        for k, r in enumerate(todo, 1):
            out("=" * 100)
            out(f"[{k}/{len(todo)}] {r['qid']}")
            out(f"Q:         {r['question']}")
            out(f"Reference: {r['reference']}")
            out(f"Answer:    {r['answer']}")
            while (choice := ask("> ").strip().lower()) not in (*LABELS, "q"):
                pass
            if choice == "q":
                break
            f.write(
                json.dumps({"pipeline": r["pipeline"], "qid": r["qid"], "label": LABELS[choice]})
                + "\n"
            )
            f.flush()
            count += 1
    return count


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(a) | set(b)) / n**2
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def agreement(rows: list[dict], labels: dict[tuple[str, str], str]) -> str | None:
    """Markdown summary of judge-human agreement, or None without labels."""
    pairs = [
        (labels[(r["pipeline"], r["qid"])], r["outcome"])
        for r in rows
        if (r["pipeline"], r["qid"]) in labels and r["outcome"] in JUDGED
    ]
    if not pairs:
        return None
    human, judge = zip(*pairs, strict=True)
    acc = sum(h == j for h, j in pairs) / len(pairs)
    kappa = cohen_kappa(list(human), list(judge))
    classes = ["correct", "partial", "incorrect"]
    lines = [
        f"{len(pairs)} answers labelled by hand. Agreement {acc:.0%}, Cohen's κ = {kappa:.2f}.",
        "",
        "| human ↓ / judge → | " + " | ".join(classes) + " |",
        "|---|" + "---|" * len(classes),
    ]
    confusion = Counter(pairs)
    for h in classes:
        lines.append(f"| {h} | " + " | ".join(str(confusion[(h, j)]) for j in classes) + " |")
    return "\n".join(lines)
