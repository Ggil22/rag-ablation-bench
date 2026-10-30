# Retrieval: query-v2

39 answerable questions. Mean with 95% bootstrap interval.
`recall@k`: share of the facts the answer needs found in the top k chunks.
`recall@1000w`: same, with the top chunks that fit in 1,000 words of context
(the fair comparison across chunk sizes). `complete@5`: all facts in the top 5.
† Retrievers that call an LLM: their wall time is dominated by free-tier throttling
or by the response cache, so it is not reported; add one LLM call to the latency.

| Chunking | Retriever | recall@5 | recall@1000w | complete@5 | mrr@10 | ms/query |
|---|---|---|---|---|---|---|
| `fixed-256` | `hybrid-rewrite-rerank` | 0.95 [0.87, 1.00] | 0.95 [0.87, 1.00] | 0.95 [0.87, 1.00] | 0.80 [0.70, 0.89] | LLM† |
| `structure-512` | `dense-hyde` | 0.97 [0.92, 1.00] | 0.94 [0.86, 1.00] | 0.97 [0.92, 1.00] | 0.86 [0.78, 0.94] | LLM† |
| `fixed-256` | `hybrid-rerank` | 0.92 [0.82, 1.00] | 0.92 [0.82, 1.00] | 0.92 [0.82, 1.00] | 0.81 [0.70, 0.90] | 610 |
| `fixed-256` | `hybrid-multi-rerank` | 0.92 [0.82, 1.00] | 0.92 [0.82, 1.00] | 0.92 [0.82, 1.00] | 0.79 [0.69, 0.89] | LLM† |
| `structure-512` | `hybrid-rerank` | 0.97 [0.92, 1.00] | 0.91 [0.82, 0.99] | 0.97 [0.92, 1.00] | 0.86 [0.77, 0.95] | 885 |
| `structure-512` | `hybrid-rewrite-rerank` | 0.97 [0.92, 1.00] | 0.91 [0.82, 0.99] | 0.97 [0.92, 1.00] | 0.88 [0.78, 0.96] | LLM† |
| `structure-512` | `hybrid-multi-rerank` | 0.97 [0.92, 1.00] | 0.91 [0.82, 0.99] | 0.97 [0.92, 1.00] | 0.86 [0.77, 0.95] | LLM† |
| `fixed-256` | `dense-hyde` | 0.92 [0.82, 1.00] | 0.90 [0.79, 0.97] | 0.92 [0.82, 1.00] | 0.74 [0.64, 0.84] | LLM† |
| `structure-512` | `hybrid-multi` | 0.92 [0.82, 1.00] | 0.87 [0.77, 0.96] | 0.92 [0.82, 1.00] | 0.82 [0.72, 0.91] | LLM† |
| `fixed-256` | `hybrid` | 0.90 [0.79, 0.97] | 0.87 [0.77, 0.97] | 0.90 [0.79, 0.97] | 0.66 [0.54, 0.78] | 2 |
| `fixed-256` | `hybrid-multi` | 0.85 [0.72, 0.95] | 0.85 [0.72, 0.95] | 0.85 [0.72, 0.95] | 0.70 [0.59, 0.81] | LLM† |
| `structure-512` | `hybrid` | 0.90 [0.79, 0.97] | 0.83 [0.72, 0.94] | 0.90 [0.79, 0.97] | 0.73 [0.61, 0.83] | 3 |
| `structure-512` | `hybrid-rewrite` | 0.91 [0.81, 0.99] | 0.81 [0.68, 0.92] | 0.90 [0.79, 0.97] | 0.74 [0.63, 0.85] | LLM† |
| `fixed-256` | `hybrid-rewrite` | 0.81 [0.68, 0.92] | 0.81 [0.68, 0.92] | 0.79 [0.67, 0.92] | 0.68 [0.56, 0.79] | LLM† |
| `structure-512` | `dense-bge-m3` | 0.82 [0.69, 0.92] | 0.79 [0.67, 0.92] | 0.82 [0.69, 0.92] | 0.76 [0.64, 0.88] | 2 |
| `fixed-256` | `dense-bge-m3` | 0.72 [0.56, 0.85] | 0.72 [0.56, 0.85] | 0.72 [0.56, 0.85] | 0.60 [0.47, 0.72] | 2 |

## Difference against `structure-512` + `hybrid`

Paired over the same questions; **bold** when the 95% interval excludes zero.

| Chunking | Retriever | Δ recall@1000w | Δ mrr@10 |
|---|---|---|---|
| `fixed-256` | `hybrid-rewrite-rerank` | **+0.12 [+0.03, +0.22]** | +0.07 [-0.04, +0.18] |
| `structure-512` | `dense-hyde` | +0.10 [+0.00, +0.23] | **+0.14 [+0.02, +0.26]** |
| `fixed-256` | `hybrid-rerank` | +0.09 [-0.03, +0.21] | +0.08 [-0.04, +0.20] |
| `fixed-256` | `hybrid-multi-rerank` | +0.09 [-0.03, +0.21] | +0.07 [-0.05, +0.18] |
| `structure-512` | `hybrid-rerank` | +0.08 [+0.00, +0.18] | **+0.14 [+0.03, +0.25]** |
| `structure-512` | `hybrid-rewrite-rerank` | +0.08 [+0.00, +0.18] | **+0.15 [+0.04, +0.26]** |
| `structure-512` | `hybrid-multi-rerank` | +0.08 [+0.00, +0.18] | **+0.14 [+0.03, +0.25]** |
| `fixed-256` | `dense-hyde` | +0.06 [-0.09, +0.22] | +0.02 [-0.14, +0.17] |
| `structure-512` | `hybrid-multi` | +0.04 [-0.06, +0.14] | +0.09 [-0.02, +0.20] |
| `fixed-256` | `hybrid` | +0.04 [-0.10, +0.18] | -0.06 [-0.17, +0.04] |
| `fixed-256` | `hybrid-multi` | +0.01 [-0.13, +0.15] | -0.02 [-0.16, +0.10] |
| `structure-512` | `hybrid-rewrite` | -0.03 [-0.18, +0.13] | +0.02 [-0.11, +0.14] |
| `fixed-256` | `hybrid-rewrite` | -0.03 [-0.18, +0.13] | -0.05 [-0.18, +0.08] |
| `structure-512` | `dense-bge-m3` | -0.04 [-0.17, +0.09] | +0.04 [-0.05, +0.14] |
| `fixed-256` | `dense-bge-m3` | -0.12 [-0.24, +0.01] | -0.13 [-0.26, +0.01] |

## recall@1000w by question type

| Chunking | Retriever | distractor (n=8) | exact_reference (n=2) | lookup (n=23) | multi_evidence (n=2) | numeric (n=4) |
|---|---|---|---|---|---|---|
| `fixed-256` | `hybrid-rewrite-rerank` | 0.88 | 1.00 | 0.96 | 1.00 | 1.00 |
| `structure-512` | `dense-hyde` | 0.88 | 1.00 | 0.96 | 0.75 | 1.00 |
| `fixed-256` | `hybrid-rerank` | 0.88 | 1.00 | 0.91 | 1.00 | 1.00 |
| `fixed-256` | `hybrid-multi-rerank` | 0.88 | 1.00 | 0.91 | 1.00 | 1.00 |
| `structure-512` | `hybrid-rerank` | 0.88 | 1.00 | 0.91 | 0.75 | 1.00 |
| `structure-512` | `hybrid-rewrite-rerank` | 0.88 | 1.00 | 0.91 | 0.75 | 1.00 |
| `structure-512` | `hybrid-multi-rerank` | 0.88 | 1.00 | 0.91 | 0.75 | 1.00 |
| `fixed-256` | `dense-hyde` | 1.00 | 1.00 | 0.87 | 1.00 | 0.75 |
| `structure-512` | `hybrid-multi` | 0.88 | 1.00 | 0.87 | 0.50 | 1.00 |
| `fixed-256` | `hybrid` | 0.88 | 1.00 | 0.83 | 1.00 | 1.00 |
| `fixed-256` | `hybrid-multi` | 1.00 | 1.00 | 0.74 | 1.00 | 1.00 |
| `structure-512` | `hybrid` | 0.75 | 1.00 | 0.83 | 0.75 | 1.00 |
| `structure-512` | `hybrid-rewrite` | 0.88 | 1.00 | 0.83 | 0.75 | 0.50 |
| `fixed-256` | `hybrid-rewrite` | 0.88 | 1.00 | 0.78 | 0.75 | 0.75 |
| `structure-512` | `dense-bge-m3` | 0.75 | 0.50 | 0.78 | 1.00 | 1.00 |
| `fixed-256` | `dense-bge-m3` | 0.62 | 0.50 | 0.74 | 1.00 | 0.75 |
