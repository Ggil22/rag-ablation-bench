# rag-ablation-bench

An engineering study of retrieval-augmented generation: **which RAG techniques actually
help, by how much, and does the answer depend on the kind of document?**

Instead of one more chatbot over PDFs, this project builds a RAG pipeline from plain
components (no framework hiding the choices) and measures every technique with an ablation
over a hand-annotated evaluation set, on three very different corpora:

| Corpus | Source | Status |
|---|---|---|
| Spanish legislation | 6 consolidated laws from the BOE open data API | ingestion + chunking done |
| Scientific papers | arXiv | planned |
| Financial reports | annual reports of IBEX companies (PDF, tables) | planned |

> **Status:** early work in progress. This README will become the technical report.

## Design decisions so far

- **Pinned, reproducible corpus.** BOE consolidated laws keep every historical version of
  each article. The corpus uses the wording in force on a fixed `as_of` date
  ([`configs/corpora.yaml`](configs/corpora.yaml)), so it does not drift when laws are
  amended.
- **No text that is not in force.** Three traps in the BOE data would let the system
  answer with law that no longer applies, and the parser handles each:
  - amendment notes quote the previous wording ("Redacción anterior") inside the article;
    they are dropped;
  - replaced blocks stay in the XML with an expiry date (`fecha_caducidad`); the TRLGDCU
    still contains a 2014 withdrawal form with a different deadline than the current one;
    expired blocks are dropped;
  - amendments can be published before their effective date is known; they are not
    treated as in force, but are counted in the section metadata (`pending_versions`).
- **Structure from content, not tags.** Some headings are tagged as articles and some
  titles use annex classes, so blocks are classified by what they contain. Provisions
  (*disposiciones adicionales, transitorias...*) have no heading block of their own and
  would otherwise inherit the last title of the law in their context header.
- **Tables kept as text.** Tax brackets and similar tables become `cell | cell` rows,
  because numeric questions depend on them.
- **Chunking-independent gold labels.** Evaluation labels point at character spans inside
  stable sections (e.g. `BOE-A-2015-11430#a37`, article 37 of the Estatuto de los
  Trabajadores). Every chunk records the exact spans it covers, so one evaluation set
  scores every chunking strategy fairly.

## Corpus: Spanish legislation

| Law | Sections | Words |
|---|---|---|
| ET — Estatuto de los Trabajadores | 142 | 56,705 |
| LPAC — Procedimiento Administrativo Común | 156 | 37,235 |
| LIRPF — Impuesto sobre la Renta de las Personas Físicas | 227 | 117,506 |
| LAU — Arrendamientos Urbanos | 79 | 24,991 |
| TRLGDCU — Defensa de los Consumidores y Usuarios | 206 | 59,541 |
| LOPDGDD — Protección de Datos Personales | 145 | 36,659 |
| **Total** | **955** | **332,637** |

Wording in force on 2026-10-30. Sections are very uneven (median 170 words, p95 1,225,
max 8,912), which is what makes chunking matter:

| Chunking | Chunks | Mean words | p95 words | Chunks mixing several articles |
|---|---|---|---|---|
| `fixed-128` | 2,973 | 128 | 128 | 30.7% |
| `fixed-256` | 1,488 | 255 | 256 | 47.8% |
| `fixed-512` | 744 | 511 | 512 | 65.2% |
| `recursive-256` | 1,561 | 213 | 254 | 29.5% |
| `recursive-512` | 750 | 444 | 509 | 48.4% |
| `structure-512` | 1,324 | 251 | 501 | 0.0% |

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run rab corpus boe     # download and parse the laws -> data/processed/boe/
uv run rab stats boe      # corpus statistics
uv run rab chunk boe      # every chunking config in configs/chunking.yaml
uv run pytest
```

## Roadmap

1. Corpus ingestion and chunking (fixed, recursive, structure-aware) ✔
2. Retrieval: BM25, dense, hybrid (RRF), cross-encoder reranking; retrieval metrics
3. Evaluation set: generated candidates, manual review, hard cases (multi-hop, exact
   references, numeric, unanswerable)
4. Generation, LLM judge validated against human labels, first ablation table
5. Papers and financial reports corpora
6. Serving (FastAPI, Qdrant), observability, p95 latency and cost per query, Docker

## Data

Legislation: © Agencia Estatal Boletín Oficial del Estado, reused under its
[open data terms](https://www.boe.es/datosabiertos/). Raw and processed data are not
committed; `rab corpus` rebuilds them.
