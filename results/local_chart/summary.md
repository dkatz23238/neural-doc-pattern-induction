# Local structural hypotheses

L1 keeps the symbol-invariant encoder and scores local facts independently: copy spans, class spans, repeat units, optional spans, alternation pairs, repeated-span pairs, and sequence cuts. Several alternatives stay alive. A chart stitches those facts into programs, and the executor checks them.

The first gate is local recall. If the gold fact is missing from the top 8 hypotheses, the recognizer is the bottleneck. If the gold facts are present and the chart's program oracle stays low, the chart is the bottleneck.

Training mixes the atomic languages with the D1 schemas. The eval compositions, including repeats of depth 4–5, are held out. BENCH-C is not scored.

Parameters: 1068179. Selected step: 1200.

| Family | Local recall@8 | Local recall@1 | Program oracle@8 | Semantic top-1 | Support acceptance | Programs |
| --- | --- | --- | --- | --- | --- | --- |
| primitive | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 8.0 |
| repeat | 1.000 | 0.567 | 0.250 | 0.000 | 0.000 | 8.0 |
| optional | 1.000 | 0.829 | 1.000 | 1.000 | 1.000 | 8.0 |
| alternation | 1.000 | 0.938 | 0.250 | 0.000 | 0.500 | 8.0 |
| prefix_repeat | 0.600 | 0.250 | 0.000 | 0.000 | 0.000 | 8.0 |
| repeat_optional | 0.622 | 0.251 | 0.000 | 0.000 | 0.000 | 8.0 |
| prefix_alt | 0.859 | 0.561 | 0.188 | 0.000 | 0.000 | 8.0 |
| optional_internal | 0.750 | 0.396 | 0.000 | 0.000 | 0.000 | 8.0 |
| repeat_extra | 1.000 | 0.567 | 1.000 | 0.000 | 0.000 | 8.0 |

| Family | Copy | Class | Repeat | Optional | Alternation | Repeat pair | Cut |
| --- | --- | --- | --- | --- | --- | --- | --- |
| primitive | 1.000 | 1.000 | — | — | — | — | 1.000 |
| repeat | 1.000 | 1.000 | 1.000 | — | — | 1.000 | 1.000 |
| optional | 1.000 | 1.000 | — | 1.000 | — | — | 1.000 |
| alternation | 1.000 | 1.000 | — | — | 1.000 | — | — |
| prefix_repeat | 1.000 | 0.000 | 0.000 | — | — | 1.000 | 1.000 |
| repeat_optional | 1.000 | 0.010 | 0.000 | 0.938 | — | 1.000 | 0.781 |
| prefix_alt | 1.000 | 1.000 | — | — | 0.438 | — | 1.000 |
| optional_internal | 1.000 | 0.000 | — | 1.000 | — | — | 1.000 |
| repeat_extra | 1.000 | 1.000 | 1.000 | — | — | 1.000 | 1.000 |

Atomic local recall@8 1.000. Held-out local recall@8 0.766. Held-out local recall@1 0.405. Held-out program oracle@8 0.237. Held-out semantic top-1 0.000. Repeat-depth local recall 1.000. Renaming invariance 1.000.

gold local structure is in the top hypotheses and the chart is not assembling a correct program.

Atomic local recall@8 is 1.000. Primitive program oracle 1.000. Optional program oracle 1.000. Repeat program oracle 0.250. Alternation program oracle 0.250.

Held-out facts absent from the top 8: prefix_repeat cls, prefix_repeat repeat, repeat_optional repeat, optional_internal cls.

Mean held-out local recall clears the recognizer line and program oracle stays low. The missing repeat units and classes above are still recognizer failures. Where the gold fact is already in the top 8, the chart is what drops the program. A pretrained backbone and BENCH-C stay off.

