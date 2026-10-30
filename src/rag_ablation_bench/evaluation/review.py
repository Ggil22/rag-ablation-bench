"""Interactive review of generated candidates: accept, edit, reject.

Decisions are saved after every candidate, so a review can stop and resume at any time.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable
from pathlib import Path

import yaml

from rag_ablation_bench.evaluation.dataset import resolve_ref
from rag_ablation_bench.retrieval.bm25 import BM25
from rag_ablation_bench.schema import Chunk, Document

CONTEXT_CHARS = 350
HELP = "[a]ccept  [e]dit then accept  [r]eject  [s]kip  [q]uit"


def save(path: Path, rows: list[dict]) -> None:
    path.write_text(
        yaml.safe_dump(rows, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
    )


def _show_evidence(item: dict, docs: list[Document], out: Callable[[str], None]) -> None:
    section = resolve_ref(item["ref"], docs)
    start = section.text.find(item["quote"])
    end = start + len(item["quote"])
    before = section.text[max(0, start - CONTEXT_CHARS) : start]
    after = section.text[end : end + CONTEXT_CHARS]
    out(f"  {item['ref']}  {section.title}")
    body = f"…{before}>>>{item['quote']}<<<{after}…"
    out(textwrap.indent(textwrap.fill(body, 100), "    "))


def _show_search(question: str, bm25: BM25, chunks: list[Chunk], out) -> None:
    out("  BM25 top 5 (confirm none of them answers the question):")
    for idx, _ in bm25.search([question], 5)[0]:
        c = chunks[idx]
        out(f"    - {c.header[-90:]}")
        out(f"      {c.text[:220].replace(chr(10), ' ')}…")


def review(
    path: Path,
    docs: list[Document],
    chunks: list[Chunk],
    *,
    ask: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
) -> dict[str, int]:
    rows = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    bm25 = None
    pending = [r for r in rows if r.get("status", "pending") == "pending"]
    out(f"{len(pending)} pending of {len(rows)} candidates. {HELP}\n")
    for n, row in enumerate(pending, 1):
        out("=" * 100)
        out(
            f"[{n}/{len(pending)}] {row['id']}  type={row['type']}  "
            f"lexical_overlap={row.get('lexical_overlap', '-')}"
        )
        out(f"Q: {row['question']}")
        out(f"A: {row['answer']}")
        if row.get("notes"):
            out(f"Notes: {row['notes']}")
        for item in row.get("evidence") or []:
            _show_evidence(item, docs, out)
        if row["type"] == "unanswerable":
            if bm25 is None:
                bm25 = BM25()
                bm25.index(chunks)
            _show_search(row["question"], bm25, chunks, out)

        while True:
            choice = ask(f"{HELP} > ").strip().lower()
            if choice in ("a", "e", "r", "s", "q"):
                break
        if choice == "q":
            break
        if choice == "s":
            continue
        if choice == "e":
            for key in ("question", "answer"):
                new = ask(f"new {key} (Enter keeps it) > ").strip()
                if new:
                    row[key] = new
        row["status"] = "rejected" if choice == "r" else "accepted"
        save(path, rows)

    counts = {
        s: sum(r.get("status", "pending") == s for r in rows)
        for s in ("accepted", "rejected", "pending")
    }
    out(f"\n{counts}")
    return counts
