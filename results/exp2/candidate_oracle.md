# Candidate oracle (CPU)

Semantic accuracy is `(hidden positive recall + hidden negative rejection) / 2`.
Candidates are generated from the visible strings only. The oracle then picks the
candidate with the highest hidden accuracy. That number is a ceiling for a ranker,
not a system we can ship: it sees the hidden set. A wider pattern can win when
the hidden negatives happen not to contain the extra strings it accepts.

The presented set has 8 candidates, heuristic first, then one slot per
mutation family. `mdl` minimizes two-part code length on the observed strings.
`rules` is the same length with an extra charge on digit literals shorter than
four characters. The linear models are ridge regressions on visible features,
with the penalty chosen on a validation split that does not include BENCH-C.

## Reading

On BENCH-C the heuristic scores 93.2%. The best of 8 symbolic neighbors scores 97.4% (+4.2 points). The full candidate pool scores 97.9%. Gold, which generated the hidden sets, scores 100.0%.

A uniform draw among those 8 neighbors scores 92.0%. Neighbors are required to match every visible string, so random selection is not a low floor. K=2 already reaches 96.0%; growing the set from 2 to 8 adds 1.4 points, and the patterns left outside the 8 add another 0.5.

The strongest CPU selector on this file is `rules` at 96.2% (+3.1 points over the heuristic). `mdl` scores 96.0%, `rules` 96.2%, the structural linear model 94.3%, and the document linear model 95.0%. That leaves 1.1 points between `rules` and the 8-candidate oracle.

Gate 1 `gpt-5.6-terra` scored 98.1% on this same file (+4.9 points over the heuristic). The 8-candidate oracle covers 85% of that gap, and the full pool is 0.2 points from Terra.

`rules` improves 14.4% of BENCH-C tasks and regresses 0.7%. The structural linear model regresses only 2.0% and leaves the heuristic unchanged on 74.7%, but it gives up accuracy to do that. The document linear model keeps the heuristic on 0.9% of tasks and regresses 28.0%.

Where the heuristic was already wrong: optional region (n=50): heuristic 85.3%, rules 88.7%, oracle@8 91.2%, pool 93.0%; variable-length inference (n=34): heuristic 81.4%, rules 82.0%, oracle@8 92.3%, pool 96.6%; character-class inference (n=27): heuristic 80.6%, rules 98.2%, oracle@8 99.0%, pool 99.5%; too narrow (n=22): heuristic 60.6%, rules 98.8%, oracle@8 99.6%, pool 100.0%; too_broad (n=10): heuristic 60.8%, rules 67.1%, oracle@8 67.1%, pool 67.1%.

Inside those families, the widest ranking gap is variable-length inference (+10.3 points between rules and oracle@8). The widest generator gap is too_broad (+32.9 points between the pool and gold).

On held-out BENCH-A shapes the heuristic scores 84.0%, `rules` 86.5%, the structural linear model 85.3%, and oracle@8 90.3% (pool 91.1%, gold 100.0%). The pool is well short of gold, so a ranker cannot close the structural-composition gap by itself.

On BENCH-D the oracle lead over `rules` is 6.5 points at 2 visible strings and 0.0 points at 20. Ranking headroom shrinks once the column is observed more times; candidate generation still matters at every sample size.

A tiny ranker is worth a pilot only as a test of the residual 1.1 points, and only if it beats `rules` on BENCH-C and on the BENCH-A shape holdout while regressing no more often than the structural linear model. The linear feature models did not beat code length, so the pilot has to use the strings themselves, not another regression on regex statistics. Failures where the pool is already near gold are ranking problems. Failures where the pool stays far from gold are generator problems and will not move if the ranker is perfect. Neither result requires a larger model or a 50k-task GPU run until that pilot wins.

## Systems

### bench_c

Held out of both linear models; primary decision set.

| System | Semantic accuracy | Macro | Recall | Rejection | Improved | Regressed | Unchanged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| heuristic | 93.2% | 92.1% | 91.4% | 94.9% | 0.0% | 0.0% | 100.0% |
| random | 92.0% | 91.2% | 92.5% | 91.5% | 7.4% | 29.8% | 15.7% |
| random_expected | 92.2% | 91.3% |  |  |  |  |  |
| mdl | 96.0% | 95.5% | 94.8% | 97.1% | 12.8% | 0.2% | 82.1% |
| rules | 96.2% | 95.7% | 95.4% | 97.1% | 14.4% | 0.7% | 78.5% |
| linear_structural | 94.3% | 93.6% | 92.3% | 96.3% | 9.4% | 2.0% | 74.7% |
| linear_document | 95.0% | 94.6% | 95.8% | 94.3% | 14.3% | 28.0% | 0.9% |
| oracle_k | 97.4% | 97.1% | 97.9% | 96.8% | 24.9% | 0.0% | 75.1% |
| oracle_pool | 97.9% | 97.9% | 98.9% | 96.9% | 29.8% | 0.0% | 70.2% |
| gold | 100.0% | 100.0% |  |  |  |  |  |

Oracle as the presented set grows (diversity order, heuristic at K=1):

| K | Tasks | Oracle | Mean pool |
| --- | --- | --- | --- |
| 1 | 554 | 93.2% |  |
| 2 | 554 | 96.0% |  |
| 4 | 554 | 96.7% |  |
| 6 | 554 | 96.9% |  |
| 8 | 554 | 97.4% |  |
| 10 | 554 | 97.6% |  |
| pool | 554 | 97.9% | 37.6 |

Oracle@8 improvements: 138 tasks, mean gain on those tasks 16.9%. Tags: no_gain 416, hex_collapse 33, slot_tight 29, width 21, slot_widen 13, optional 10, width_aggressive 8, affix 7, class_widen 6, repeat 6, catalog 2, literal_digit 2, class_tighten 1.

### bench_d

Held out; same schemas at several sample sizes.

| System | Semantic accuracy | Macro | Recall | Rejection | Improved | Regressed | Unchanged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| heuristic | 89.8% | 89.8% | 85.1% | 94.6% | 0.0% | 0.0% | 100.0% |
| random | 90.3% | 90.3% | 88.5% | 92.0% | 12.5% | 32.2% | 16.2% |
| random_expected | 89.8% | 89.8% |  |  |  |  |  |
| mdl | 92.2% | 92.2% | 86.3% | 98.0% | 10.6% | 0.9% | 87.2% |
| rules | 93.8% | 93.8% | 89.6% | 97.9% | 16.6% | 0.0% | 82.2% |
| linear_structural | 93.5% | 93.5% | 89.8% | 97.2% | 15.3% | 8.4% | 57.2% |
| linear_document | 91.0% | 91.0% | 90.9% | 91.1% | 15.9% | 42.5% | 0.0% |
| oracle_k | 96.8% | 96.8% | 96.3% | 97.2% | 29.4% | 0.0% | 70.6% |
| oracle_pool | 97.4% | 97.4% | 97.4% | 97.4% | 29.7% | 0.0% | 70.3% |
| gold | 100.0% | 100.0% |  |  |  |  |  |

Oracle as the presented set grows (diversity order, heuristic at K=1):

| K | Tasks | Oracle | Mean pool |
| --- | --- | --- | --- |
| 1 | 320 | 89.8% |  |
| 2 | 320 | 92.4% |  |
| 4 | 320 | 95.5% |  |
| 6 | 320 | 95.5% |  |
| 8 | 320 | 96.8% |  |
| 10 | 320 | 97.2% |  |
| pool | 320 | 97.4% | 24.7 |

Oracle@8 improvements: 94 tasks, mean gain on those tasks 23.5%. Tags: no_gain 226, hex_collapse 28, slot_tight 16, width_aggressive 14, width 13, optional 10, slot_widen 9, literal_digit 2, slot_digit_general 2.

### bench_a_test

Held-out AST shapes.

| System | Semantic accuracy | Macro | Recall | Rejection | Improved | Regressed | Unchanged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| heuristic | 84.0% | 84.0% | 86.1% | 82.0% | 0.0% | 0.0% | 100.0% |
| random | 84.0% | 84.0% | 88.0% | 79.9% | 16.5% | 30.5% | 21.0% |
| random_expected | 83.9% | 83.9% |  |  |  |  |  |
| mdl | 86.5% | 86.5% | 85.5% | 87.5% | 21.5% | 4.5% | 69.5% |
| rules | 86.5% | 86.5% | 85.8% | 87.3% | 23.0% | 6.5% | 64.0% |
| linear_structural | 85.3% | 85.3% | 87.7% | 82.9% | 21.5% | 15.0% | 47.0% |
| linear_document | 84.5% | 84.5% | 91.1% | 77.9% | 29.5% | 35.0% | 8.5% |
| oracle_k | 90.3% | 90.3% | 94.8% | 85.8% | 41.5% | 0.0% | 58.5% |
| oracle_pool | 91.1% | 91.1% | 96.2% | 86.1% | 47.5% | 0.0% | 52.5% |
| gold | 100.0% | 100.0% |  |  |  |  |  |

Oracle as the presented set grows (diversity order, heuristic at K=1):

| K | Tasks | Oracle | Mean pool |
| --- | --- | --- | --- |
| 1 | 200 | 84.0% |  |
| 2 | 200 | 87.1% |  |
| 4 | 200 | 89.7% |  |
| 6 | 200 | 90.0% |  |
| 8 | 200 | 90.3% |  |
| 10 | 200 | 90.4% |  |
| pool | 200 | 91.1% | 19.1 |

Oracle@8 improvements: 83 tasks, mean gain on those tasks 15.2%. Tags: no_gain 117, slot_tight 38, class_widen 20, affix 7, width 6, slot_widen 4, width_aggressive 4, class_tighten 2, hex_collapse 1, literal_digit 1.

### bench_a_val

Penalty selection for the structural linear model.

| System | Semantic accuracy | Macro | Recall | Rejection | Improved | Regressed | Unchanged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| heuristic | 84.0% | 84.0% | 89.8% | 78.1% | 0.0% | 0.0% | 100.0% |
| random | 84.4% | 84.4% | 90.8% | 77.9% | 22.5% | 29.0% | 20.0% |
| random_expected | 85.0% | 85.0% |  |  |  |  |  |
| mdl | 88.3% | 88.3% | 89.5% | 87.2% | 31.0% | 5.0% | 58.5% |
| rules | 88.4% | 88.4% | 89.8% | 87.1% | 31.0% | 5.5% | 56.5% |
| linear_structural | 86.0% | 86.0% | 90.0% | 82.0% | 29.5% | 10.5% | 46.5% |
| linear_document | 84.5% | 84.5% | 91.6% | 77.5% | 33.5% | 36.0% | 6.5% |
| oracle_k | 91.1% | 91.1% | 95.8% | 86.4% | 46.0% | 0.0% | 54.0% |
| oracle_pool | 91.9% | 91.9% | 96.9% | 86.9% | 51.0% | 0.0% | 49.0% |
| gold | 100.0% | 100.0% |  |  |  |  |  |

Oracle as the presented set grows (diversity order, heuristic at K=1):

| K | Tasks | Oracle | Mean pool |
| --- | --- | --- | --- |
| 1 | 200 | 84.0% |  |
| 2 | 200 | 88.2% |  |
| 4 | 200 | 90.3% |  |
| 6 | 200 | 90.5% |  |
| 8 | 200 | 91.1% |  |
| 10 | 200 | 91.3% |  |
| pool | 200 | 91.9% | 17.9 |

Oracle@8 improvements: 92 tasks, mean gain on those tasks 15.5%. Tags: no_gain 108, slot_tight 34, class_widen 25, slot_widen 13, affix 8, width 5, width_aggressive 4, optional 2, class_tighten 1.

### bench_a_dev

Training data for the structural linear model.

| System | Semantic accuracy | Macro | Recall | Rejection | Improved | Regressed | Unchanged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| heuristic | 84.6% | 84.6% | 88.3% | 80.9% | 0.0% | 0.0% | 100.0% |
| random | 85.2% | 85.2% | 89.9% | 80.5% | 18.2% | 30.9% | 18.0% |
| random_expected | 84.9% | 84.9% |  |  |  |  |  |
| mdl | 87.6% | 87.6% | 87.6% | 87.6% | 21.9% | 4.0% | 67.9% |
| rules | 87.7% | 87.7% | 87.9% | 87.5% | 22.6% | 5.2% | 65.0% |
| linear_structural | 86.4% | 86.4% | 89.4% | 83.5% | 23.5% | 12.2% | 47.0% |
| linear_document | 84.7% | 84.7% | 91.1% | 78.4% | 28.1% | 36.0% | 6.6% |
| oracle_k | 90.8% | 90.8% | 95.3% | 86.2% | 43.1% | 0.0% | 56.9% |
| oracle_pool | 91.7% | 91.7% | 96.7% | 86.6% | 50.1% | 0.0% | 49.9% |
| gold | 100.0% | 100.0% |  |  |  |  |  |

Oracle as the presented set grows (diversity order, heuristic at K=1):

| K | Tasks | Oracle | Mean pool |
| --- | --- | --- | --- |
| 1 | 800 | 84.6% |  |
| 2 | 800 | 88.0% |  |
| 4 | 800 | 90.0% |  |
| 6 | 800 | 90.4% |  |
| 8 | 800 | 90.8% |  |
| 10 | 800 | 91.0% |  |
| pool | 800 | 91.7% | 19.2 |

Oracle@8 improvements: 345 tasks, mean gain on those tasks 14.4%. Tags: no_gain 455, slot_tight 121, class_widen 100, slot_widen 40, width_aggressive 31, width 18, affix 10, optional 10, class_tighten 8, hex_collapse 5, literal_digit 2.

### bench_b_val

Penalty selection for the document linear model.

| System | Semantic accuracy | Macro | Recall | Rejection | Improved | Regressed | Unchanged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| heuristic | 91.9% | 90.6% | 90.9% | 92.9% | 0.0% | 0.0% | 100.0% |
| random | 90.9% | 89.6% | 91.2% | 90.6% | 6.6% | 25.5% | 17.2% |
| random_expected | 91.5% | 90.4% |  |  |  |  |  |
| mdl | 95.4% | 95.2% | 95.1% | 95.8% | 16.6% | 0.3% | 76.8% |
| rules | 95.9% | 95.6% | 96.1% | 95.7% | 19.9% | 0.3% | 73.5% |
| linear_structural | 94.3% | 93.5% | 93.2% | 95.5% | 15.9% | 0.3% | 81.1% |
| linear_document | 95.2% | 94.9% | 96.4% | 94.0% | 20.5% | 16.2% | 0.0% |
| oracle_k | 96.8% | 96.5% | 97.8% | 95.7% | 26.8% | 0.0% | 73.2% |
| oracle_pool | 96.9% | 96.6% | 98.1% | 95.7% | 26.8% | 0.0% | 73.2% |
| gold | 100.0% | 100.0% |  |  |  |  |  |

Oracle as the presented set grows (diversity order, heuristic at K=1):

| K | Tasks | Oracle | Mean pool |
| --- | --- | --- | --- |
| 1 | 302 | 91.9% |  |
| 2 | 302 | 95.3% |  |
| 4 | 302 | 96.5% |  |
| 6 | 302 | 96.5% |  |
| 8 | 302 | 96.8% |  |
| 10 | 302 | 96.8% |  |
| pool | 302 | 96.9% | 37.9 |

Oracle@8 improvements: 81 tasks, mean gain on those tasks 18.2%. Tags: no_gain 221, hex_collapse 25, class_widen 23, slot_tight 14, width 12, optional 3, repeat 2, slot_widen 1, width_aggressive 1.

### bench_b_train

Training data for the document linear model.

| System | Semantic accuracy | Macro | Recall | Rejection | Improved | Regressed | Unchanged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| heuristic | 91.3% | 89.3% | 88.2% | 94.4% | 0.0% | 0.0% | 100.0% |
| random | 91.3% | 89.8% | 90.5% | 92.2% | 9.2% | 23.6% | 15.6% |
| random_expected | 91.2% | 89.6% |  |  |  |  |  |
| mdl | 95.3% | 94.8% | 93.8% | 96.8% | 14.8% | 0.5% | 78.3% |
| rules | 96.1% | 95.3% | 95.5% | 96.7% | 19.0% | 0.0% | 74.5% |
| linear_structural | 93.2% | 91.7% | 90.2% | 96.2% | 11.5% | 0.0% | 84.6% |
| linear_document | 95.5% | 94.7% | 96.1% | 94.9% | 19.9% | 14.9% | 0.0% |
| oracle_k | 97.2% | 96.6% | 97.8% | 96.7% | 26.6% | 0.0% | 73.4% |
| oracle_pool | 97.5% | 97.0% | 98.4% | 96.7% | 27.3% | 0.0% | 72.7% |
| gold | 100.0% | 100.0% |  |  |  |  |  |

Oracle as the presented set grows (diversity order, heuristic at K=1):

| K | Tasks | Oracle | Mean pool |
| --- | --- | --- | --- |
| 1 | 1198 | 91.3% |  |
| 2 | 1198 | 95.4% |  |
| 4 | 1198 | 96.9% |  |
| 6 | 1198 | 96.9% |  |
| 8 | 1198 | 97.2% |  |
| 10 | 1198 | 97.4% |  |
| pool | 1198 | 97.5% | 39.8 |

Oracle@8 improvements: 319 tasks, mean gain on those tasks 22.3%. Tags: no_gain 879, hex_collapse 100, slot_tight 69, class_widen 54, width 46, repeat 22, slot_widen 14, width_aggressive 7, affix 4, optional 3.

## BENCH-C by heuristic failure

`rank_gap` is oracle@8 minus the hand-written rule. `truncation_gap` is the full pool minus oracle@8. `generator_gap` is gold minus the full pool.

| Failure | n | Heuristic | MDL | Rules | Linear doc | Oracle@8 | Pool | Gold | Rank gap | Truncation | Generator |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| success | 402 | 99.1% | 99.1% | 99.1% | 97.7% | 99.3% | 99.5% | 100.0% | +0.2 | +0.1 | +0.5 |
| optional region | 50 | 85.3% | 87.4% | 88.7% | 87.1% | 91.2% | 93.0% | 100.0% | +2.5 | +1.8 | +7.0 |
| variable-length inference | 34 | 81.4% | 81.4% | 82.0% | 83.0% | 92.3% | 96.6% | 100.0% | +10.3 | +4.2 | +3.4 |
| character-class inference | 27 | 80.6% | 96.0% | 98.2% | 96.1% | 99.0% | 99.5% | 100.0% | +0.8 | +0.5 | +0.5 |
| too narrow | 22 | 60.6% | 98.8% | 98.8% | 98.8% | 99.6% | 100.0% | 100.0% | +0.9 | +0.4 | +0.0 |
| too_broad | 10 | 60.8% | 67.1% | 67.1% | 65.0% | 67.1% | 67.1% | 100.0% | +0.0 | +0.0 | +32.9 |
| mixed population | 4 | 63.5% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 100.0% | +0.0 | +0.0 | +8.8 |
| too broad | 3 | 68.1% | 70.8% | 70.8% | 70.8% | 70.8% | 70.8% | 100.0% | +0.0 | +0.0 | +29.2 |
| OCR noise | 1 | 79.2% | 79.2% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | +0.0 | +0.0 | +0.0 |
| alternation | 1 | 89.6% | 89.6% | 89.6% | 85.4% | 89.6% | 89.6% | 100.0% | +0.0 | +0.0 | +10.4 |

## BENCH-D sample size

| Visible strings | n | Heuristic | Rules | Oracle@8 | Pool |
| --- | --- | --- | --- | --- | --- |
| 2 | 64 | 81.1% | 85.9% | 92.4% | 94.5% |
| 3 | 64 | 86.2% | 90.1% | 96.1% | 97.2% |
| 5 | 64 | 92.5% | 95.8% | 97.9% | 97.9% |
| 10 | 64 | 94.0% | 98.1% | 98.6% | 98.6% |
| 20 | 64 | 95.3% | 98.8% | 98.8% | 98.8% |

## Protocol

Linear structural model trained on BENCH-A `dev` (800 tasks), penalty chosen on BENCH-A val (λ=0.1, val accuracy 86.0%).

Linear document model trained on BENCH-B fold `bench_b hash % 5 != 0` (1198 tasks), penalty chosen on the held-out BENCH-B fold (λ=100.0, val accuracy 95.2%).

BENCH-C and BENCH-D are not used to fit either model or to choose λ. Mean candidate-stage latency on BENCH-C is 5.0 ms (p50 2.6 ms).

Terra's BENCH-C accuracy is the Gate 1 figure in results/exp1 (gpt-5.6-terra, simple prompt, 2026-09-25). This run does not call a model API.
