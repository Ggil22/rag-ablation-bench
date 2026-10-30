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
RETRIEVAL_CONFIG = Path("configs/retrieval.yaml")
MODELS_CONFIG = Path("configs/models.yaml")
PIPELINES_CONFIG = Path("configs/pipelines.yaml")
DEFAULT_QUESTIONS = [Path("evalsets/boe/seed.yaml")]
DATA = Path("data")
RESULTS = Path("results")
BUILDERS = {"boe": boe.build}
# Naive RAG: fixed-size chunks and an embedding model. Every other system is compared to it.
DEFAULT_BASELINE = "fixed-256:dense-bge-m3"


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


def _llm_factory():
    """name -> LLM client, created lazily so commands that need no API need no keys."""
    from functools import cache

    from dotenv import load_dotenv

    from rag_ablation_bench.llm import LLM, MissingAPIKeyError, load_model_configs

    load_dotenv(Path(".env"))
    configs = load_model_configs(MODELS_CONFIG)

    @cache
    def make(name: str) -> LLM:
        if name not in configs:
            sys.exit(f"Unknown model {name!r}. See {MODELS_CONFIG}.")
        try:
            return LLM(configs[name])
        except MissingAPIKeyError as exc:
            sys.exit(str(exc))

    return make


def _llm_retrievers() -> frozenset[str]:
    return frozenset(k for k, v in _load_yaml(RETRIEVAL_CONFIG).items() if v.get("llm"))


def _questions(paths: list[Path], docs):
    from rag_ablation_bench.evaluation.dataset import InvalidQuestions, load_question_sets

    try:
        return load_question_sets(paths, docs)
    except InvalidQuestions as exc:
        sys.exit(f"Invalid questions:\n{exc}")


def _chunks(docs, chunk_configs: dict, name: str):
    params = dict(chunk_configs[name])
    return chunk_documents(docs, params.pop("strategy"), **params)


def _select(requested: list[str] | None, available: dict, what: str, path: Path) -> list[str]:
    names = requested or list(available)
    unknown = [n for n in names if n not in available]
    if unknown:
        sys.exit(f"Unknown {what}: {', '.join(unknown)}. See {path}.")
    return names


def cmd_eval_retrieval(args: argparse.Namespace) -> None:
    from rag_ablation_bench.evaluation import retrieval_eval
    from rag_ablation_bench.retrieval import RetrieverFactory

    chunk_configs = _load_yaml(CHUNKING_CONFIG)
    retrieval_configs = _load_yaml(RETRIEVAL_CONFIG)
    chunkings = _select(args.chunking, chunk_configs, "chunking configs", CHUNKING_CONFIG)
    # The default grid leaves out retrievers that call an LLM API; name them to include them.
    available = retrieval_configs
    if not args.retrievers:
        available = {k: v for k, v in retrieval_configs.items() if not v.get("llm")}
    retrievers = _select(args.retrievers, available, "retrievers", RETRIEVAL_CONFIG)

    docs = _load_corpus(args.corpus)
    questions = _questions(args.questions, docs)
    chunk_sets = {name: _chunks(docs, chunk_configs, name) for name in chunkings}

    out_dir = RESULTS / "retrieval" / args.run_id
    rows = retrieval_eval.run(
        chunk_sets,
        retrievers,
        RetrieverFactory(retrieval_configs, llm=_llm_factory()),
        questions,
        out_dir / "per_question.jsonl",
    )
    report = retrieval_eval.write_report(
        rows,
        out_dir / "report.md",
        title=f"Retrieval: {args.corpus}, {' + '.join(p.name for p in args.questions)}",
        baseline=tuple(args.baseline.split(":", 1)),
        llm_retrievers=_llm_retrievers(),
    )
    print(f"\nReport: {report}")


def cmd_report_retrieval(args: argparse.Namespace) -> None:
    from rag_ablation_bench.evaluation import retrieval_eval

    rows = retrieval_eval.load_rows(args.run_dir / "per_question.jsonl")
    report = retrieval_eval.write_report(
        rows,
        args.run_dir / "report.md",
        title=f"Retrieval: {args.run_dir.name}",
        baseline=tuple(args.baseline.split(":", 1)),
        llm_retrievers=_llm_retrievers(),
    )
    print(report)


def cmd_gen_questions(args: argparse.Namespace) -> None:
    from rag_ablation_bench.evaluation import question_gen
    from rag_ablation_bench.evaluation.review import save

    if args.out.exists() and not args.force:
        sys.exit(f"{args.out} exists; review it, choose another --out or pass --force.")
    docs = _load_corpus(args.corpus)
    # Prefer sections the existing sets do not cover yet.
    existing = _questions(args.exclude, docs) if args.exclude else []
    used = {s.section_id for q in existing for f in q.facts for s in f.spans}
    chunks = _chunks(docs, _load_yaml(CHUNKING_CONFIG), "structure-512")
    candidates, stats = question_gen.generate(
        docs,
        chunks,
        _llm_factory()(args.model),
        n_single=args.single,
        n_linked=args.linked,
        n_unanswerable=args.unanswerable,
        used_sections=used,
        seed=args.seed,
    )
    rows = [
        {"id": f"{args.prefix}-{i:03d}", **c, "source": args.model, "status": "pending"}
        for i, c in enumerate(candidates, 1)
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save(args.out, rows)
    print(
        f"{'kind':<14} {'requested':>9} {'kept':>5} {'skipped':>8} {'misquoted':>10} {'failed':>7}"
    )
    for kind in ("single", "linked", "unanswerable"):
        print(
            f"{kind:<14} {stats.requested[kind]:>9} {stats.kept[kind]:>5} "
            f"{stats.skipped[kind]:>8} {stats.misquoted[kind]:>10} {stats.failed[kind]:>7}"
        )
    print(f"\n{len(rows)} candidates in {args.out}. Review them with `rab review {args.out}`.")


def cmd_review(args: argparse.Namespace) -> None:
    from rag_ablation_bench.evaluation.review import review

    docs = _load_corpus(args.corpus)
    review(args.path, docs, _chunks(docs, _load_yaml(CHUNKING_CONFIG), "structure-512"))


def _e2e_paths(run_id: str) -> tuple[Path, Path, Path]:
    out_dir = RESULTS / "e2e" / run_id
    return out_dir / "per_question.jsonl", out_dir / "report.md", out_dir / "human_labels.jsonl"


def _e2e_report(rows: list[dict], run_id: str, config: dict) -> Path:
    from rag_ablation_bench.evaluation import e2e
    from rag_ablation_bench.evaluation.labeling import agreement, load_labels

    _, report, labels = _e2e_paths(run_id)
    return e2e.write_report(
        rows,
        report,
        title=f"End-to-end: {run_id}",
        generator=_llm_factory()(config["generator"]),
        baseline=config["baseline"],
        agreement=agreement(rows, load_labels(labels)),
    )


def cmd_eval_e2e(args: argparse.Namespace) -> None:
    from rag_ablation_bench.evaluation import e2e
    from rag_ablation_bench.evaluation.judge import Judge
    from rag_ablation_bench.llm import QuotaExhausted
    from rag_ablation_bench.rag import RAG
    from rag_ablation_bench.retrieval import RetrieverFactory

    config = _load_yaml(PIPELINES_CONFIG)
    pipelines = _select(args.pipelines, config["pipelines"], "pipelines", PIPELINES_CONFIG)
    make_llm = _llm_factory()
    generator, judge = make_llm(config["generator"]), Judge(make_llm(config["judge"]))
    factory = RetrieverFactory(_load_yaml(RETRIEVAL_CONFIG), llm=make_llm)
    chunk_configs = _load_yaml(CHUNKING_CONFIG)

    docs = _load_corpus(args.corpus)
    questions = _questions(args.questions, docs)
    results, _, _ = _e2e_paths(args.run_id)
    results.parent.mkdir(parents=True, exist_ok=True)
    # Step 1: answers. Resumable: questions a pipeline already answered are skipped.
    try:
        for name in pipelines:
            spec = config["pipelines"][name]
            rag = None
            if not spec.get("closed_book"):
                chunks = _chunks(docs, chunk_configs, spec["chunking"])
                retriever = factory(spec["retriever"])
                retriever.index(chunks)
                retriever.search(["calentamiento"], 1)  # load models before timing
                rag = RAG(retriever, chunks, generator, budget=config["budget"])
            e2e.generate_answers(name, rag, generator, questions, results)
    except QuotaExhausted as exc:
        sys.exit(f"Generator stopped: {exc}\nRerun the same command later to continue.")
    # Step 2: verdicts.
    _judge(args.run_id, config, judge)


def _judge(run_id: str, config: dict, judge) -> None:
    from rag_ablation_bench.evaluation import e2e

    results, _, _ = _e2e_paths(run_id)
    rows = e2e.load_rows(results)
    pending = e2e.judge_rows(rows, judge, results)
    if pending:
        print(f"{pending} rows still to judge: run `rab judge {run_id}` later.")
    print(f"\nReport: {_e2e_report(rows, run_id, config)}")


def cmd_judge(args: argparse.Namespace) -> None:
    from rag_ablation_bench.evaluation.judge import Judge

    config = _load_yaml(PIPELINES_CONFIG)
    _judge(args.run_id, config, Judge(_llm_factory()(config["judge"])))


def cmd_report_e2e(args: argparse.Namespace) -> None:
    from rag_ablation_bench.evaluation import e2e

    rows = e2e.load_rows(_e2e_paths(args.run_id)[0])
    print(_e2e_report(rows, args.run_id, _load_yaml(PIPELINES_CONFIG)))


def cmd_label(args: argparse.Namespace) -> None:
    from rag_ablation_bench.evaluation import e2e
    from rag_ablation_bench.evaluation.labeling import label

    results, _, labels = _e2e_paths(args.run_id)
    label(e2e.load_rows(results), labels, n=args.n)
    print(f"Labels in {labels}. Rebuild the report with `rab report-e2e {args.run_id}`.")


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

    p = sub.add_parser("eval-retrieval", help="evaluate chunking x retriever on a question set")
    p.add_argument("corpus", choices=list(BUILDERS))
    p.add_argument("--questions", type=Path, nargs="+", default=DEFAULT_QUESTIONS)
    p.add_argument("--chunking", nargs="+", help=f"names from {CHUNKING_CONFIG} (default: all)")
    p.add_argument("--retrievers", nargs="+", help=f"names from {RETRIEVAL_CONFIG} (default: all)")
    p.add_argument("--run-id", required=True, help="results go to results/retrieval/<run-id>/")
    p.add_argument("--baseline", default=DEFAULT_BASELINE, help="chunking:retriever")
    p.set_defaults(func=cmd_eval_retrieval)

    p = sub.add_parser("report-retrieval", help="rebuild the report of a retrieval run")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--baseline", default=DEFAULT_BASELINE, help="chunking:retriever")
    p.set_defaults(func=cmd_report_retrieval)

    p = sub.add_parser("gen-questions", help="generate candidate questions with an LLM")
    p.add_argument("corpus", choices=list(BUILDERS))
    # Neither the generator nor the judge: it should not answer or grade its own questions,
    # and on free tiers it should not spend their quota.
    p.add_argument("--model", default="groq-gpt-oss-20b")
    p.add_argument("--single", type=int, default=120, help="single-passage questions")
    p.add_argument("--linked", type=int, default=30, help="questions needing two articles")
    p.add_argument("--unanswerable", type=int, default=20)
    p.add_argument(
        "--exclude",
        type=Path,
        nargs="*",
        default=DEFAULT_QUESTIONS,
        help="question sets whose sections should not be reused",
    )
    p.add_argument("--out", type=Path, default=Path("evalsets/boe/generated.yaml"))
    p.add_argument("--prefix", default="gen")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="overwrite --out")
    p.set_defaults(func=cmd_gen_questions)

    p = sub.add_parser("review", help="accept, edit or reject generated questions")
    p.add_argument("path", type=Path)
    p.add_argument("--corpus", choices=list(BUILDERS), default="boe")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("eval-e2e", help="retrieval + generation + judge, per pipeline")
    p.add_argument("corpus", choices=list(BUILDERS))
    p.add_argument("--questions", type=Path, nargs="+", default=DEFAULT_QUESTIONS)
    p.add_argument("--pipelines", nargs="+", help=f"names from {PIPELINES_CONFIG} (default: all)")
    p.add_argument("--run-id", required=True, help="results go to results/e2e/<run-id>/")
    p.set_defaults(func=cmd_eval_e2e)

    p = sub.add_parser("judge", help="judge the pending answers of an end-to-end run")
    p.add_argument("run_id")
    p.set_defaults(func=cmd_judge)

    p = sub.add_parser("report-e2e", help="rebuild the report of an end-to-end run")
    p.add_argument("run_id")
    p.set_defaults(func=cmd_report_e2e)

    p = sub.add_parser("label", help="label answers by hand to validate the LLM judge")
    p.add_argument("run_id")
    p.add_argument("-n", type=int, default=60, help="total answers to label")
    p.set_defaults(func=cmd_label)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
