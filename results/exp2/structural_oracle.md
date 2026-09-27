# Structural candidate generators

Five generator families are added to the local neighborhood: span and class merging, alignment via shared affixes, delimiter and repeat structure, mixture alternation, and a short code-length beam over grammar edits. A candidate still has to match every visible string. The oracle scores the hidden set. The code-length rule sees only the visible strings. BENCH-C is not used to choose a generator.

## Reading

On BENCH-C the previous oracle@8 scores 97.4%. The structural pool reaches oracle@8 99.1%, oracle@16 99.5%, and a full-pool oracle of 99.6%. The code-length rule on that pool scores 95.4%, against 96.2% on the old 8 (regressions 6.1% versus 0.7%).

Repeated groups, optional spans, and sentinel alternation put the right regex in the first 8 to 16 candidates. Mean pool size stays under 50. K=64 does not move the oracle past K=32. Candidate generation is the right direction on document fields.

Too-broad fields go from oracle 67.1% to 99.6%, and the rule reaches 86.2%; variable-length oracle reaches 99.8% while the rule stays at 83.8%; too-narrow fields, already solved by the old rule, fall to rule 71.4% against oracle 100.0%.

The pool contains the better regex. The rule does not take it. Literals are free in the cost model, so a pattern that clings to the visible strings can be shorter than the pattern that survives the hidden set.

On held-out BENCH-A shapes the oracle only moves from 90.3% to 91.7%. The rule there scores 85.7% against 86.5% on the old 8. Deep synthetic alternations are still outside this generator.

## C

Curated document fields. Not used to design the generators beyond the failure list.

| System | Semantic accuracy | Macro | Improved | Regressed | Unchanged |
| --- | --- | --- | --- | --- | --- |
| heuristic | 93.2% | 92.1% | 0.0% | 0.0% | 100.0% |
| rules on the old 8 | 96.2% | 95.7% | 14.4% | 0.7% | 84.8% |
| oracle@8, old pool | 97.4% | 97.1% | 24.9% | 0.0% | 75.1% |
| oracle, old pool | 97.9% | 97.9% | 29.8% | 0.0% | 70.2% |
| rules@8, structural order | 95.4% | 94.8% | 17.5% | 6.0% | 76.5% |
| oracle@8, structural order | 99.1% | 99.0% | 35.4% | 0.0% | 64.6% |
| oracle@16 | 99.5% | 99.5% | 39.2% | 0.0% | 60.8% |
| oracle@32 | 99.6% | 99.6% | 39.7% | 0.0% | 60.3% |
| oracle@64 | 99.6% | 99.6% | 40.1% | 0.0% | 59.9% |
| rules on the full structural pool | 95.4% | 94.9% | 20.4% | 6.1% | 73.5% |
| oracle on the full structural pool | 99.6% | 99.6% | 40.1% | 0.0% | 59.9% |

Mean pool size 37.6 before, 48.7 after. Mean latency 53.2 ms.

Heuristic-failure families:

| Failure | n | heuristic | old oracle@8 | oracle@32 | oracle pool | rules pool |
| --- | --- | --- | --- | --- | --- | --- |
| success | 402 | 99.1% | 99.3% | 99.7% | 99.7% | 98.7% |
| optional region | 50 | 85.3% | 91.2% | 98.2% | 98.2% | 93.3% |
| variable-length inference | 34 | 81.4% | 92.3% | 99.8% | 99.8% | 83.8% |
| character-class inference | 27 | 80.6% | 99.0% | 99.5% | 99.5% | 91.9% |
| too narrow | 22 | 60.6% | 99.6% | 100.0% | 100.0% | 71.4% |
| too_broad | 10 | 60.8% | 67.1% | 99.6% | 99.6% | 86.2% |
| mixed population | 4 | 63.5% | 91.1% | 100.0% | 100.0% | 85.4% |
| too broad | 3 | 68.1% | 70.8% | 100.0% | 100.0% | 65.3% |
| OCR noise | 1 | 79.2% | 100.0% | 100.0% | 100.0% | 100.0% |
| alternation | 1 | 89.6% | 89.6% | 100.0% | 100.0% | 100.0% |

Tag of the best pattern in the structural pool:

| Tag | Tasks | Share |
| --- | --- | --- |
| heuristic | 178 | 32.1% |
| ast_search | 90 | 16.2% |
| width_aggressive | 88 | 15.9% |
| slot_tight | 32 | 5.8% |
| hex_collapse | 29 | 5.2% |
| segment | 29 | 5.2% |
| delimiter_split | 17 | 3.1% |
| width | 16 | 2.9% |
| grouping | 15 | 2.7% |
| optional_span | 13 | 2.3% |
| repeat_motif | 12 | 2.2% |
| span_merge | 11 | 2.0% |

## A-test

Held-out AST shapes.

| System | Semantic accuracy | Macro | Improved | Regressed | Unchanged |
| --- | --- | --- | --- | --- | --- |
| heuristic | 84.0% | 84.0% | 0.0% | 0.0% | 100.0% |
| rules on the old 8 | 86.5% | 86.5% | 23.0% | 6.5% | 70.5% |
| oracle@8, old pool | 90.3% | 90.3% | 41.5% | 0.0% | 58.5% |
| oracle, old pool | 91.1% | 91.1% | 47.5% | 0.0% | 52.5% |
| rules@8, structural order | 85.8% | 85.8% | 25.5% | 9.5% | 65.0% |
| oracle@8, structural order | 90.7% | 90.7% | 44.5% | 0.0% | 55.5% |
| oracle@16 | 91.4% | 91.4% | 48.0% | 0.0% | 52.0% |
| oracle@32 | 91.6% | 91.6% | 49.5% | 0.0% | 50.5% |
| oracle@64 | 91.7% | 91.7% | 49.5% | 0.0% | 50.5% |
| rules on the full structural pool | 85.7% | 85.7% | 26.5% | 10.5% | 63.0% |
| oracle on the full structural pool | 91.7% | 91.7% | 49.5% | 0.0% | 50.5% |

Mean pool size 19.1 before, 24.2 after. Mean latency 25.3 ms.

Tag of the best pattern in the structural pool:

| Tag | Tasks | Share |
| --- | --- | --- |
| heuristic | 48 | 24.0% |
| ast_search | 33 | 16.5% |
| slot_tight | 33 | 16.5% |
| class_widen | 30 | 15.0% |
| width_aggressive | 19 | 9.5% |
| width | 11 | 5.5% |
| affix | 7 | 3.5% |
| slot_widen | 7 | 3.5% |
| segment | 4 | 2.0% |
| mixture | 3 | 1.5% |
| span_merge | 2 | 1.0% |
| class_tighten | 1 | 0.5% |

## A-dev

BENCH-A development shapes, reported so the test split is not the only structural number.

| System | Semantic accuracy | Macro | Improved | Regressed | Unchanged |
| --- | --- | --- | --- | --- | --- |
| heuristic | 84.6% | 84.6% | 0.0% | 0.0% | 100.0% |
| rules on the old 8 | 87.7% | 87.7% | 22.6% | 5.2% | 72.1% |
| oracle@8, old pool | 90.8% | 90.8% | 43.1% | 0.0% | 56.9% |
| oracle, old pool | 91.7% | 91.7% | 50.1% | 0.0% | 49.9% |
| rules@8, structural order | 86.8% | 86.8% | 24.0% | 9.0% | 67.0% |
| oracle@8, structural order | 91.5% | 91.5% | 47.1% | 0.0% | 52.9% |
| oracle@16 | 92.1% | 92.1% | 50.5% | 0.0% | 49.5% |
| oracle@32 | 92.3% | 92.3% | 51.6% | 0.0% | 48.4% |
| oracle@64 | 92.3% | 92.3% | 52.4% | 0.0% | 47.6% |
| rules on the full structural pool | 86.7% | 86.7% | 26.2% | 9.9% | 63.9% |
| oracle on the full structural pool | 92.3% | 92.3% | 52.4% | 0.0% | 47.6% |

Mean pool size 19.2 before, 24.3 after. Mean latency 27.3 ms.

Tag of the best pattern in the structural pool:

| Tag | Tasks | Share |
| --- | --- | --- |
| ast_search | 188 | 23.5% |
| heuristic | 155 | 19.4% |
| class_widen | 122 | 15.2% |
| slot_tight | 88 | 11.0% |
| width_aggressive | 86 | 10.8% |
| width | 46 | 5.8% |
| slot_widen | 44 | 5.5% |
| mixture | 18 | 2.2% |
| segment | 12 | 1.5% |
| class_tighten | 10 | 1.2% |
| affix | 9 | 1.1% |
| optional | 9 | 1.1% |

## D

Same schemas at several sample sizes.

| System | Semantic accuracy | Macro | Improved | Regressed | Unchanged |
| --- | --- | --- | --- | --- | --- |
| heuristic | 89.8% | 89.8% | 0.0% | 0.0% | 100.0% |
| rules on the old 8 | 93.8% | 93.8% | 16.6% | 0.0% | 83.4% |
| oracle@8, old pool | 96.8% | 96.8% | 29.4% | 0.0% | 70.6% |
| oracle, old pool | 97.4% | 97.4% | 29.7% | 0.0% | 70.3% |
| rules@8, structural order | 90.2% | 90.2% | 19.1% | 12.2% | 68.8% |
| oracle@8, structural order | 96.2% | 96.2% | 37.5% | 0.0% | 62.5% |
| oracle@16 | 97.8% | 97.8% | 40.3% | 0.0% | 59.7% |
| oracle@32 | 98.0% | 98.0% | 40.3% | 0.0% | 59.7% |
| oracle@64 | 98.1% | 98.1% | 40.3% | 0.0% | 59.7% |
| rules on the full structural pool | 90.9% | 90.9% | 19.7% | 12.5% | 67.8% |
| oracle on the full structural pool | 98.1% | 98.1% | 40.3% | 0.0% | 59.7% |

Mean pool size 24.7 before, 34.1 after. Mean latency 19.2 ms.

Tag of the best pattern in the structural pool:

| Tag | Tasks | Share |
| --- | --- | --- |
| width_aggressive | 99 | 30.9% |
| heuristic | 95 | 29.7% |
| hex_collapse | 33 | 10.3% |
| delimiter_split | 22 | 6.9% |
| slot_tight | 15 | 4.7% |
| span_merge | 12 | 3.8% |
| ast_search | 11 | 3.4% |
| optional | 9 | 2.8% |
| width | 9 | 2.8% |
| slot_widen | 5 | 1.6% |
| literal_digit | 3 | 0.9% |
| repeat_motif | 3 | 0.9% |
