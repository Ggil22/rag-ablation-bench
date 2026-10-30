# rag-ablation-bench

An engineering study of retrieval-augmented generation over Spanish legislation: **which RAG
techniques actually help, by how much, and at what cost?**

Instead of one more chatbot over PDFs, the pipeline is built from plain components (no
framework hiding the choices) and every technique is measured with an ablation over an
annotated evaluation set, with confidence intervals, on a corpus with real traps.

## Key findings (retrieval, 39 answerable questions)

- **Hybrid search is the single biggest win over the tutorial baseline.** Adding BM25 to
  dense retrieval with reciprocal rank fusion raises recall at equal context from 0.72 to
  0.87 (+0.15, 95% CI [+0.05, +0.28]). Legal questions mix colloquial wording, where
  embeddings help, with exact terms such as *alzada* or *fianza*, where BM25 does.
- **The best configuration finds the evidence for 91–92% of the facts**, against 72% for
  the naive pipeline (fixed 256-word chunks + embeddings): hybrid search + cross-encoder
  reranking.
- **HyDE matches the cross-encoder without a GPU.** Asking an LLM for a hypothetical
  article and searching with it reaches 0.94 recall, the same as reranking (difference
  +0.03, CI [−0.05, +0.10]), at the price of one LLM call per query instead of ~0.6–0.9 s
  of GPU time.
- **Query rewriting and multi-query do not help** here (−0.03 and +0.04, both within noise),
  despite being standard advice.
- **Comparing chunk sizes by recall@k is misleading.** With BM25, 512-word chunks beat
  128-word chunks at recall@5 (0.77 vs 0.49) only because five large chunks carry four
  times more text; at an equal budget of 1,000 words the order flips (0.49 vs 0.56). All
  chunking comparisons here use an equal word budget.
- **Once a reranker is in place, chunking matters less**: fixed 256-word and
  structure-aware chunks tie (0.92 vs 0.91), while 512-word fixed chunks lose (0.79).

End to end (partial, see [below](#end-to-end-results)): **retrieval failures turn into
abstentions rather than wrong answers**, and without retrieval the model answers 3 of 4
questions the corpus cannot source, while every RAG pipeline abstains.

## Corpus: Spanish legislation

Six consolidated laws from the BOE open data API, with the wording in force on 2026-10-30:

| Law | Sections | Words |
|---|---|---|
| ET — Estatuto de los Trabajadores | 142 | 56,705 |
| LPAC — Procedimiento Administrativo Común | 156 | 37,235 |
| LIRPF — Impuesto sobre la Renta de las Personas Físicas | 227 | 117,506 |
| LAU — Arrendamientos Urbanos | 79 | 24,991 |
| TRLGDCU — Defensa de los Consumidores y Usuarios | 206 | 59,541 |
| LOPDGDD — Protección de Datos Personales | 145 | 36,659 |
| **Total** | **955** | **332,637** |

Getting this right took more than downloading text:

- **Pinned, reproducible corpus.** Consolidated laws keep every historical version of each
  article; the corpus uses the version in force on a fixed `as_of` date
  ([`configs/corpora.yaml`](configs/corpora.yaml)), so it does not drift as laws change.
- **No text that is not in force.** Three traps in the BOE data would let the system
  answer with law that no longer applies:
  - amendment notes quote the previous wording ("Redacción anterior") inside the article;
  - replaced blocks stay in the XML with an expiry date (`fecha_caducidad`); the consumer
    law still contains a withdrawn form with a different deadline than the current one;
  - amendments can be published before their effective date is known (LAU art. 10, in
    October 2026).

  The parser drops the first two and records the third as `pending_versions`.
- **Structure from content, not tags.** Some headings are tagged as articles and some
  titles use annex markup; provisions (*disposiciones adicionales, transitorias...*) have no
  heading block and would otherwise inherit the last title of the law in their context.
- **Tables kept as text** (`cell | cell` rows): numeric questions such as tax brackets
  depend on them.

## Evaluation methodology

**Question set.** [`evalsets/boe/seed.yaml`](evalsets/boe/seed.yaml): 43 questions written
by reading the laws, in plain language rather than legal wording, of six types: plain
lookups, numeric (often in tables), exact references to an article, **distractors** (a
similar provision gives a wrong answer: appeal deadlines in two near-identical articles,
the €40,000 threshold that appears both for lottery prizes and for data-protection fines,
transitional regimes with superseded figures), multi-evidence questions and 4
**unanswerable** ones that look answerable (the minimum wage amount, the tax-return
deadline: both set by other regulations).

**Gold labels are exact quotes**, resolved to character spans inside stable section ids and
checked against the corpus on load. Every chunk records the spans it covers, so one set of
labels scores every chunking strategy. A fact counts as found when the retrieved chunks
cover at least half of its span.

**Fair comparison across chunk sizes.** Besides recall@k, every configuration is scored on
the first 1,000 words of retrieved context (the last chunk truncated), so large chunks
cannot win by handing over more text.

**Statistics.** Means with 95% bootstrap intervals; differences between systems are paired
over the same questions and called significant only when the interval excludes zero. With
39 answerable questions, differences under ~0.10 are usually noise, and the tables say so.

**Annotation audit.** Error analysis of the first run found one gold-label omission (the
withdrawal period is also stated in an annex). It was fixed and documented in the question
(`v2`); both runs are kept ([`seed-v1`](results/retrieval/seed-v1/report.md),
[`seed-v2`](results/retrieval/seed-v2/report.md)).

**Scaling the set.** `rab gen-questions` drafts candidates with an LLM, each with a quote
verified against the corpus (candidates with invented quotes are dropped and counted), and
records the lexical overlap between question and evidence, because generated questions
tend to reuse the law's wording and so favour BM25. They do: the first 122 candidates
([`generated.yaml`](evalsets/boe/generated.yaml)) share a median 44% of their content words
with the evidence, against 17% for the hand-written questions, even though the prompt asks
for everyday wording. 4% of candidates were dropped for quoting text that is not in the law. `rab review` is the human pass:
accept, edit or reject, with the quote shown in context and, for unanswerable candidates,
what the search engine finds. Candidates are not filtered by how hard they are for the
systems under test, which would bias the set against them.

## Retrieval results

The ablation path from the naive pipeline, one change at a time (recall on the first
1,000 words of context; full tables in [`seed-v2`](results/retrieval/seed-v2/report.md) and
[`query-v2`](results/retrieval/query-v2/report.md)):

| Step | Chunking | Retriever | Recall@1000w | Δ vs previous step | MRR@10 | Latency |
|---|---|---|---|---|---|---|
| Naive | fixed 256 | dense (bge-m3) | 0.72 [0.56, 0.85] | | 0.60 | 2 ms |
| + BM25 (RRF) | fixed 256 | hybrid | 0.87 [0.77, 0.97] | **+0.15 [+0.05, +0.28]** | 0.66 | 2 ms |
| + cross-encoder | fixed 256 | hybrid + rerank | 0.92 [0.82, 1.00] | +0.05 [−0.05, +0.15] | 0.81 | 610 ms |
| Structure chunks | structure 512 | hybrid + rerank | 0.91 [0.82, 0.99] | −0.01 [−0.09, +0.06] | 0.86 | 885 ms |
| HyDE instead of reranker | structure 512 | dense + HyDE | 0.94 [0.86, 1.00] | +0.03 vs above | 0.86 | 1 LLM call |

Latency is per query on an RTX 4060 (reranking 50 candidates with bge-reranker-v2-m3 in
fp16). The reranker's gain shows mostly in ranking quality: MRR rises from 0.66 to 0.81.

Other ablations (structure-aware chunks unless noted):

| Question | Result |
|---|---|
| Which embedding model? | Qwen3-Embedding-0.6B 0.82 ≈ bge-m3 0.79 > multilingual-e5-base 0.74 (overlapping CIs) |
| Spanish stemming in BM25? | MRR +0.03 to +0.08 on all four chunkings tested; each within noise, consistent in sign |
| Prefix chunks with their location ("ET > Título I > Artículo 37")? | No measurable effect, for BM25 or dense |
| LLM query rewriting? | −0.03 [−0.18, +0.13]; on top of the reranker, +0.03 |
| Multi-query (3 reformulations + RRF)? | +0.04 [−0.06, +0.14] |
| Chunking, with reranking | fixed 256: 0.92, structure: 0.91, recursive 256: 0.85, fixed 128: 0.87, recursive 512: 0.82, fixed 512: 0.79 |

**Where retrieval still fails** (best configuration):
- *"¿Cuántas semanas de permiso tiene el padre por el nacimiento de un hijo?"*: the
  retrievers prefer the transitional provisions describing the 2019–2021 phase-in (8 and 12
  weeks) over article 48 (19 weeks). Superseded figures that are still in the law are the
  hardest distractor.
- *"Compré una lavadora hace dos años y ha dejado de funcionar..."*: everyday wording
  ("dejado de funcionar", "sigue respondiendo") against legal wording ("falta de
  conformidad"); 61 of 76 retrieval runs miss article 120, and all 18 with a cross-encoder
  do: the reranker pushes it out even when the first stage found it.

## End-to-end results

> **In progress.** Run [`seed-v2`](results/e2e/seed-v2/) has 162 of 258 answers (6
> pipelines × 43 questions) and the LLM judge has scored few of them: both free tiers ran
> out of daily quota (Gemini for generation, Groq's 200k tokens/day for judging). The run
> resumes where it stopped. Below are only the metrics that need no judge, which are
> complete for the answers generated so far.

Generator `gemini-3.5-flash-lite`, 2,000 words of retrieved context:

| Pipeline | Answered an unanswerable question | Abstained on an answerable one | Evidence in context | Retrieval p50 | Generation p50 / p95 |
|---|---|---|---|---|---|
| closed-book (no retrieval) | 3 of 4 | 0 of 39 | – | – | 3.7 s / 25.7 s |
| naive (fixed 256, dense) | 0 of 4 | 4 of 39 | 87% | 79 ms | 5.8 s / 19.9 s |
| hybrid (fixed 256) | 0 of 4 | 2 of 39 | 92% | 79 ms | 4.7 s / 21.3 s |
| hybrid + rerank (33 of 43 so far) | – | 1 of 33 | 97% | 885 ms | 6.1 s / 37.2 s |

What these already show:

- **Retrieval failures become abstentions, not wrong answers.** All 7 abstentions on
  answerable questions, across pipelines, happened when the retrieved context lacked the
  evidence (context recall 0); no pipeline abstained when the evidence was there. Better
  retrieval therefore converts directly into answered questions.
- **Without retrieval the model answers what it cannot source.** Closed-book answered 3 of
  the 4 questions the corpus cannot answer (VAT rate, sick-leave duration, tax-return
  deadline). Its answers are roughly right in the real world, but they come from the
  model's memory, with no article to cite and no way to tell when that memory is out of
  date; the RAG pipelines abstained on all four.
- **Generation dominates latency.** Retrieval costs 79 ms (≈0.9 s with the cross-encoder)
  against seconds of generation on a free-tier API, whose p95 is 4–6× its median.

Accuracy against the reference answers, faithfulness of the cited passages, the remaining
pipelines (structure-aware chunks, HyDE) and the validation of the judge against human
labels will be added when the run completes.

## Serving

`rag_ablation_bench.serve.app` serves one pipeline from
[`configs/pipelines.yaml`](configs/pipelines.yaml) over HTTP, reusing the exact `RAG` object
that was evaluated:

```bash
uv sync --extra models --extra serve
uv run uvicorn rag_ablation_bench.serve.app:app --port 8080
curl -X POST localhost:8080/ask -H "Content-Type: application/json" \
     -d '{"question": "¿Cuántos días de vacaciones me corresponden?"}'
```

The response carries the answer, the cited articles with links to the BOE, and the
retrieval and generation times, token usage and cost. Every request is appended to
`logs/requests.jsonl`; with `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` set it is also
traced in Langfuse, with spans opened around the real retrieval and generation work.
`scripts/loadtest.py` measures p50/p95/p99 latency and throughput under concurrency.

`docker compose up --build` builds the corpus into the image and runs the API on the GPU
(NVIDIA Container Toolkit required). The image has not been built yet on the development
machine.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/); GPU optional but recommended for dense retrieval
and reranking. API keys go in `.env` (see [`configs/models.yaml`](configs/models.yaml)).

```bash
uv sync --extra models
uv run rab corpus boe                       # download and parse the laws
uv run rab chunk boe                        # chunking statistics
uv run rab eval-retrieval boe --run-id my-run            # offline retrieval grid
uv run rab eval-retrieval boe --run-id q --retrievers dense-hyde hybrid-multi  # LLM-based
uv run rab eval-e2e boe --run-id my-e2e     # generation + judge, all pipelines
uv run rab judge my-e2e                     # resume judging after a quota limit
uv run rab label my-e2e                     # hand-label answers to validate the judge
uv run rab gen-questions boe && uv run rab review evalsets/boe/generated.yaml
uv run pytest
```

LLM calls and embeddings are cached on disk, so re-running a command only pays for what
changed, and long runs that hit a provider's daily quota stop cleanly and resume.

## Limitations

- **Small question set.** 43 questions give intervals of roughly ±0.10–0.15; only large
  effects are conclusive. Generated candidates are waiting for human review.
- **One corpus.** Scientific papers and financial reports are planned, to test whether the
  conclusions transfer.
- **One annotator.** The seed questions were written and labelled by one person.

## Roadmap

1. ✔ Corpus ingestion and chunking (fixed, recursive, structure-aware)
2. ✔ Retrieval: BM25, dense, hybrid, reranking, query transformation; retrieval metrics
3. ◐ Evaluation set: seed set done; generated candidates awaiting review
4. ◐ Generation and LLM judge; judge validation against human labels pending
5. Papers and financial reports corpora
6. ◐ Serving, observability, load testing; Docker image to be built and measured

## Data

Legislation: © Agencia Estatal Boletín Oficial del Estado, reused under its
[open data terms](https://www.boe.es/datosabiertos/). Raw and processed data are not
committed; `rab corpus` rebuilds them.
