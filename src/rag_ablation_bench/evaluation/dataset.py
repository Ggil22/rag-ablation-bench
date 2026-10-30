"""Evaluation questions with gold evidence given as exact quotes from the corpus.

Quotes, not character offsets, are what annotators write: they are readable, survive small
edits to the file, and are checked against the corpus on load. Each quote becomes a Span.

    - id: boe-001
      question: ¿Cuántos días libres me corresponden si me caso?
      answer: Quince días naturales.
      type: lookup
      evidence:                       # every item is a fact the answer needs
        - ref: "ET: Artículo 37"      # "<short title>: <section title>" or "<short>#<block id>"
          quote: Quince días naturales en caso de matrimonio
          alternatives:               # optional: other places stating the same fact
            - ref: ...
              quote: ...

A question with no evidence is unanswerable from the corpus. Generated candidates carry a
`status` and only the accepted ones are loaded.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from rag_ablation_bench.schema import Document, Section, Span


@dataclass(frozen=True)
class Fact:
    """One piece of evidence the answer needs; finding any of its spans is enough."""

    spans: tuple[Span, ...]


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    answer: str
    type: str
    facts: tuple[Fact, ...]
    notes: str = ""

    @property
    def answerable(self) -> bool:
        return bool(self.facts)


class InvalidQuestions(ValueError):
    pass


def resolve_ref(ref: str, docs: list[Document]) -> Section:
    """Find a section by "<short>#<block id>" or by "<short>: <title>", where the title may
    stop at the number ("LOPDGDD: Artículo 22" matches "Artículo 22. Tratamientos ...")."""
    sep = "#" if "#" in ref else ":"
    short, key = (part.strip() for part in ref.split(sep, 1))
    doc = next((d for d in docs if d.short_title == short), None)
    if doc is None:
        raise KeyError(f"unknown document {short!r}")
    if sep == "#":
        matches = [s for s in doc.sections if s.id == f"{doc.id}#{key}"]
    else:
        # Match up to the dot, so "Artículo 48" does not also match "Artículo 48 bis".
        matches = [s for s in doc.sections if s.title == key or s.title.startswith(f"{key}.")]
    if len(matches) != 1:
        raise KeyError(f"{ref!r} matches {len(matches)} sections")
    return matches[0]


def load_questions(path: Path, docs: list[Document]) -> list[Question]:
    errors: list[str] = []

    def span(qid: str, item: dict) -> Span | None:
        try:
            section = resolve_ref(item["ref"], docs)
        except KeyError as exc:
            errors.append(f"{qid}: {exc.args[0]}")
            return None
        start = section.text.find(item["quote"])
        if start < 0:
            errors.append(f"{qid}: quote not found in {item['ref']}: {item['quote'][:80]!r}")
            return None
        return Span(section.id, start, start + len(item["quote"]))

    questions, seen = [], set()
    for row in yaml.safe_load(path.read_text(encoding="utf-8")):
        if row.get("status", "accepted") != "accepted":  # generated, not yet reviewed
            continue
        qid = row["id"]
        if qid in seen:
            errors.append(f"{qid}: duplicate id")
        seen.add(qid)
        facts = []
        for item in row.get("evidence") or []:
            spans = [span(qid, i) for i in [item, *item.get("alternatives", [])]]
            facts.append(Fact(tuple(s for s in spans if s is not None)))
        questions.append(
            Question(
                id=qid,
                question=row["question"],
                answer=str(row["answer"]),
                type=row["type"],
                facts=tuple(facts),
                notes=row.get("notes", ""),
            )
        )
    if errors:
        raise InvalidQuestions("\n".join(errors))
    return questions


def load_question_sets(paths: list[Path], docs: list[Document]) -> list[Question]:
    questions = [q for path in paths for q in load_questions(path, docs)]
    dupes = sorted(qid for qid, n in Counter(q.id for q in questions).items() if n > 1)
    if dupes:
        raise InvalidQuestions(f"ids repeated across files: {', '.join(dupes)}")
    return questions
