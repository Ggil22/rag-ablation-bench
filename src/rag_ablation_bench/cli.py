"""Command line interface: `rab <command>`."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

from rag_ablation_bench.chunking import STRATEGIES, chunk_documents, chunk_stats
from rag_ablation_bench.corpus import boe
from rag_ablation_bench.schema import load_documents, save_chunks, save_documents

CORPORA_CONFIG = Path("configs/corpora.yaml")
CHUNKING_CONFIG = Path("configs/chunking.yaml")
DATA = Path("data")
BUILDERS = {"boe": boe.build}


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _documents_path(corpus: str) -> Path:
    return DATA / "processed" / corpus / "documents.jsonl"


def _load_corpus(corpus: str):
    path = _documents_path(corpus)
    if not path.exists():
        sys.exit(f"{path} not found. Run `rab corpus {corpus}` first.")
    return load_documents(path)


def cmd_corpus(args: argparse.Namespace) -> None:
    config = _load_yaml(CORPORA_CONFIG)[args.corpus]
    raw_dir = DATA / "raw" / args.corpus
    docs = BUILDERS[args.corpus](config, raw_dir, force_download=args.force_download)
    save_documents(docs, _documents_path(args.corpus))
    for d in docs:
        print(f"{d.short_title:<9} {len(d.sections):>5} sections {d.word_count():>8,} words")
    print(f"\nWrote {_documents_path(args.corpus)}")


def cmd_stats(args: argparse.Namespace) -> None:
    docs = _load_corpus(args.corpus)
    sections = [s for d in docs for s in d.sections]
    sizes = sorted(len(s.text.split()) for s in sections)
    kinds = Counter(s.kind for s in sections)
    repealed = sum(s.metadata.get("repealed", False) for s in sections)
    dupes = sum(n - 1 for n in Counter(s.text for s in sections).values() if n > 1)
    print(f"documents:  {len(docs)}")
    print(f"sections:   {len(sections)}  {dict(kinds)}")
    print(f"words:      {sum(sizes):,}")
    print(
        f"words/sec:  median {sizes[len(sizes) // 2]}, p95 {sizes[int(len(sizes) * 0.95)]}, "
        f"max {sizes[-1]}"
    )
    print(f"repealed:   {repealed}")
    print(f"duplicates: {dupes} sections with the exact text of an earlier one")
    print("\nLongest sections:")
    for s in sorted(sections, key=lambda s: -len(s.text.split()))[: args.top]:
        print(f"  {len(s.text.split()):>6}  {s.id:<32} {s.title[:60]}")


def cmd_chunk(args: argparse.Namespace) -> None:
    configs = _load_yaml(CHUNKING_CONFIG)
    names = args.configs or list(configs)
    unknown = [n for n in names if n not in configs]
    if unknown:
        sys.exit(f"Unknown chunking configs: {', '.join(unknown)}. See {CHUNKING_CONFIG}.")
    docs = _load_corpus(args.corpus)

    header = ("config", "chunks", "mean", "p50", "p95", "max", "multi-sec")
    print("{:<16} {:>7} {:>6} {:>5} {:>5} {:>5} {:>9}".format(*header))
    for name in names:
        params = dict(configs[name])
        chunks = chunk_documents(docs, params.pop("strategy"), **params)
        save_chunks(chunks, DATA / "processed" / args.corpus / "chunks" / f"{name}.jsonl")
        st = chunk_stats(chunks)
        print(
            f"{name:<16} {st['chunks']:>7} {st['mean_words']:>6.0f} {st['p50_words']:>5} "
            f"{st['p95_words']:>5} {st['max_words']:>5} {st['multi_section']:>9.1%}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("corpus", help="download and parse a corpus")
    p.add_argument("corpus", choices=list(BUILDERS))
    p.add_argument("--force-download", action="store_true", help="ignore cached raw files")
    p.set_defaults(func=cmd_corpus)

    p = sub.add_parser("stats", help="describe a built corpus")
    p.add_argument("corpus", choices=list(BUILDERS))
    p.add_argument("--top", type=int, default=10, help="how many of the longest sections")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("chunk", help=f"chunk a corpus ({', '.join(STRATEGIES)})")
    p.add_argument("corpus", choices=list(BUILDERS))
    p.add_argument("configs", nargs="*", help=f"names from {CHUNKING_CONFIG} (default: all)")
    p.set_defaults(func=cmd_chunk)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
