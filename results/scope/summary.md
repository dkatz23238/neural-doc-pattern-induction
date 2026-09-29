# Scope relations

P2b freezes the version-space chart and the L1 recognizer. New heads read the generator's groups: same repeated unit, containment, group start and end, optional child, alternation branch, and shared prefix or suffix. The training objective is recall at 8. Each strong class or repeat span keeps a second bound, and alternation is the frozen bilinear plus a residual.

The checkpoint is chosen on atomic local recall. Held-out local recall and the version-space oracle on the model's own top 8 are measured after that choice. BENCH-C is not scored.

Parameters: 1123387 (55208 trained). Selected step: 800.

| Family | Local recall@8 | Local recall@1 | Version-space oracle | Open-repeat oracle | Programs | Truncated |
| --- | --- | --- | --- | --- | --- | --- |
| primitive | 1.000 | 1.000 | 1.000 | 1.000 | 808.1 | 0 |
| repeat | 1.000 | 0.567 | 1.000 | 1.000 | 672.1 | 0 |
| optional | 1.000 | 0.829 | 1.000 | 1.000 | 156.1 | 0 |
| alternation | 1.000 | 1.000 | 1.000 | 0.000 | 126.0 | 0 |
| prefix_repeat | 1.000 | 0.250 | 1.000 | 1.000 | 1034.4 | 0 |
| repeat_optional | 0.952 | 0.251 | 0.938 | 0.938 | 8.2 | 0 |
| prefix_alt | 1.000 | 0.561 | 1.000 | 0.438 | 176.5 | 0 |
| optional_internal | 1.000 | 0.396 | 1.000 | 1.000 | 1779.6 | 0 |
| repeat_extra | 1.000 | 0.567 | 1.000 | 1.000 | 927.2 | 0 |

| Family | Copy | Class | Repeat | Optional | Alternation | Repeat pair | Cut |
| --- | --- | --- | --- | --- | --- | --- | --- |
| primitive | 1.000 | 1.000 | — | — | — | — | 1.000 |
| repeat | 1.000 | 1.000 | 1.000 | — | — | 1.000 | 1.000 |
| optional | 1.000 | 1.000 | — | 1.000 | — | — | 1.000 |
| alternation | 1.000 | 1.000 | — | — | 1.000 | — | — |
| prefix_repeat | 1.000 | 1.000 | 1.000 | — | — | 1.000 | 1.000 |
| repeat_optional | 1.000 | 0.995 | 1.000 | 0.938 | — | 1.000 | 0.781 |
| prefix_alt | 1.000 | 1.000 | — | — | 1.000 | — | 1.000 |
| optional_internal | 1.000 | 1.000 | — | 1.000 | — | — | 1.000 |
| repeat_extra | 1.000 | 1.000 | 1.000 | — | — | 1.000 | 1.000 |

Atomic local recall@8 1.000. Atomic version-space oracle 1.000. Held-out local recall@8 0.990. Held-out local recall@1 0.405. Held-out version-space oracle 0.988. Renaming invariance 1.000. Truncated searches: 0.

held-out local recall cleared 0.95 and the version-space oracle recovered programs from the model's own facts.

Local recall and the version-space oracle both cleared their lines on the model's own facts. Held-out tasks leave 785 probe-consistent programs on average. Choosing among those programs is the following experiment. A pretrained backbone and BENCH-C stay off.

