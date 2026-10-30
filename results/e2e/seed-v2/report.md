# End-to-end: seed-v2

> **Incomplete:** 133 answers still to be judged (`rab judge`).

Generator `gemini-3.5-flash-lite`. 10 answerable and 4 unanswerable questions. Accuracy: correct = 1, partial = 0.5, correct abstention = 1; mean with 95% bootstrap interval.

| Pipeline | n | Accuracy | Correct (answerable) | Hallucinated (unanswerable) | False abstention | Faithful | Context recall | p50 / p95 latency | Tokens / query |
|---|---|---|---|---|---|---|---|---|---|
| `closed-book` | 14 | 0.79 [0.57, 1.00] | 100% | 75% | 0% | – | – | 3.6 s / 5.9 s | 209 |
| `hybrid` | 6 | 0.67 [0.33, 1.00] | 0% | 0% | 100% | – | 0% | 6.6 s / 77.6 s | 3,438 |
| `naive` | 8 | 0.50 [0.12, 0.88] | 0% | 0% | 100% | – | 0% | 9.1 s / 23.4 s | 3,414 |
| `hybrid-rerank` | 1 | 0.00 [0.00, 0.00] | 0% | – | 100% | – | 0% | 6.9 s / 6.9 s | 3,327 |

## Accuracy difference against `naive`

Paired over the same questions; **bold** when the 95% interval excludes zero.

| Pipeline | Δ accuracy | Questions |
|---|---|---|
| `closed-book` | -0.40 [-1.00, +0.40] | 5 |
| `hybrid` | +0.00 [+0.00, +0.00] | 6 |
| `hybrid-rerank` | +nan [+nan, +nan] | 0 |

## Outcomes

| Pipeline | correct | partial | incorrect | false_abstention | correct_abstention | hallucinated | error |
|---|---|---|---|---|---|---|---|
| `closed-book` | 10 | 0 | 0 | 0 | 1 | 3 | 0 |
| `hybrid` | 0 | 0 | 0 | 2 | 4 | 0 | 0 |
| `naive` | 0 | 0 | 0 | 4 | 4 | 0 | 0 |
| `hybrid-rerank` | 0 | 0 | 0 | 1 | 0 | 0 | 0 |

## Accuracy by question type

| Pipeline | distractor | lookup | numeric | unanswerable |
|---|---|---|---|---|
| `closed-book` | 1.00 | 1.00 | 1.00 | 0.25 |
| `hybrid` | 0.00 | 0.00 | nan | 1.00 |
| `naive` | 0.00 | 0.00 | nan | 1.00 |
| `hybrid-rerank` | nan | 0.00 | nan | nan |

## Failures of the best pipeline (`closed-book`)

**boe-041** (unanswerable, hallucinated): ¿Hasta qué día se puede presentar la declaración de la renta?
- Reference: UNANSWERABLE
- Answer: El plazo de presentación de la declaración del Impuesto sobre la Renta de las Personas Físicas (IRPF) finaliza habitualmente el 30 de junio del año siguiente al periodo impositivo correspondiente (o el 25 de junio si el resultado es a ingresar con domiciliación bancaria), según se establece en la Orden ministerial anual que aprueba los modelos de declaración.

**boe-042** (unanswerable, hallucinated): ¿Cuánto dura como máximo una baja por incapacidad temporal?
- Reference: UNANSWERABLE
- Answer: La duración máxima de la incapacidad temporal es de 365 días, prorrogables por otros 180 días cuando se presuma que durante ellos puede el trabajador ser dado de alta médica por curación o mejoría.

**boe-043** (unanswerable, hallucinated): ¿Cuál es el tipo general del IVA en España?
- Reference: UNANSWERABLE
- Answer: El tipo general del Impuesto sobre el Valor Añadido (IVA) en España es del 21%.

