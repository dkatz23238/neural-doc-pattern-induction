# Experiment 1 — symbolic baselines

Semantic accuracy is `(positive recall + negative rejection) / 2` on hidden examples.
Latency is wall time on this cloud VM CPU, not a laptop.

Frontier baselines are part of Gate 1 but were not executed in this run because neither OPENAI_API_KEY nor ANTHROPIC_API_KEY is set.

## bench_c

| System | Semantic accuracy | Macro accuracy | Positive recall | Negative rejection | Valid | Catastrophic | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| memorize | 51.7% | 51.9% | 3.4% | 100.0% | 100.0% | 0.0% | 0.01 | 0.02 |
| heuristic | 93.2% | 92.1% | 91.4% | 94.9% | 100.0% | 3.2% | 0.23 | 0.82 |
| grex_default | 51.7% | 51.9% | 3.4% | 100.0% | 100.0% | 0.0% | 2.07 | 5.51 |
| grex_digits | 77.0% | 75.7% | 54.3% | 99.6% | 100.0% | 0.0% | 1.57 | 4.47 |
| grex_words | 82.9% | 81.8% | 90.0% | 75.8% | 100.0% | 0.0% | 1.43 | 2.04 |
| grex_repetitions | 51.7% | 51.9% | 3.4% | 100.0% | 100.0% | 0.0% | 2.11 | 5.94 |
| grex_combined | 86.6% | 84.1% | 82.9% | 90.3% | 100.0% | 0.0% | 1.53 | 3.24 |
| fast | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 3931.07 | 4624.18 |

`fast` returned no expression on 8 of 8 tasks (accuracy on completed runs: None). A zero in the table is a timeout, not a scored regex.

Failure taxonomy:

- `memorize`: variable-length inference 230, too narrow 225, optional region 69, success 15, OCR noise 10, alternation 5
- `heuristic`: success 402, optional region 49, variable-length inference 34, character-class inference 24, too narrow 22, too_broad 18
- `grex_default`: variable-length inference 230, too narrow 225, optional region 69, success 15, OCR noise 10, alternation 5
- `grex_digits`: success 251, variable-length inference 139, too narrow 130, optional region 34
- `grex_words`: character-class inference 209, optional region 151, variable-length inference 109, success 52, too broad 20, mixed population 5
- `grex_repetitions`: variable-length inference 230, too narrow 225, optional region 69, success 15, OCR noise 10, alternation 5
- `grex_combined`: success 281, variable-length inference 86, optional region 74, character-class inference 54, too narrow 43, too broad 12
- `fast`: invalid_output 8

## bench_b

| System | Semantic accuracy | Macro accuracy | Positive recall | Negative rejection | Valid | Catastrophic | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| memorize | 50.0% | 50.0% | 0.0% | 100.0% | 100.0% | 0.0% | 0.01 | 0.02 |
| heuristic | 91.4% | 89.5% | 88.7% | 94.1% | 100.0% | 6.6% | 0.21 | 1.43 |
| grex_default | 50.0% | 50.0% | 0.0% | 100.0% | 100.0% | 0.0% | 1.95 | 6.55 |
| grex_digits | 80.5% | 74.7% | 62.0% | 99.0% | 100.0% | 0.0% | 1.55 | 5.33 |
| grex_words | 83.7% | 82.9% | 91.1% | 76.3% | 100.0% | 0.0% | 1.46 | 2.02 |
| grex_repetitions | 50.0% | 50.0% | 0.0% | 100.0% | 100.0% | 0.0% | 2.18 | 9.08 |
| grex_combined | 86.5% | 83.7% | 80.3% | 92.8% | 100.0% | 0.0% | 1.53 | 4.82 |
| fast | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 3089.12 | 6008.55 |

`fast` returned no expression on 5 of 8 tasks (accuracy on completed runs: None). A zero in the table is a timeout, not a scored regex.

Failure taxonomy:

- `memorize`: too narrow 750, variable-length inference 500, optional region 250
- `heuristic`: success 1137, too_broad 99, too narrow 93, variable-length inference 81, optional region 58, character-class inference 31
- `grex_default`: too narrow 750, variable-length inference 500, optional region 250
- `grex_digits`: success 850, too narrow 375, variable-length inference 266, optional region 9
- `grex_words`: character-class inference 610, variable-length inference 352, optional region 308, success 211, semantic class recognition 9, mixed population 5
- `grex_repetitions`: too narrow 750, variable-length inference 500, optional region 250
- `grex_combined`: success 952, variable-length inference 279, too narrow 125, character-class inference 122, optional region 22
- `fast`: invalid_output 8

## bench_d

| System | Semantic accuracy | Macro accuracy | Positive recall | Negative rejection | Valid | Catastrophic | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| memorize | 50.0% | 50.0% | 0.0% | 100.0% | 100.0% | 0.0% | 0.01 | 0.03 |
| heuristic | 89.8% | 89.8% | 85.1% | 94.6% | 100.0% | 5.9% | 0.20 | 0.59 |
| grex_default | 50.0% | 50.0% | 0.0% | 100.0% | 100.0% | 0.0% | 1.78 | 4.41 |
| grex_digits | 67.9% | 67.9% | 36.1% | 99.7% | 100.0% | 0.0% | 1.54 | 2.54 |
| grex_words | 80.9% | 80.9% | 89.1% | 72.8% | 100.0% | 0.0% | 1.43 | 1.73 |
| grex_repetitions | 50.0% | 50.0% | 0.0% | 100.0% | 100.0% | 0.0% | 1.73 | 4.11 |
| grex_combined | 85.7% | 85.7% | 82.5% | 88.9% | 100.0% | 0.0% | 1.26 | 1.64 |
| fast | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 3095.66 | 3394.27 |

`fast` returned no expression on 8 of 8 tasks (accuracy on completed runs: None). A zero in the table is a timeout, not a scored regex.

Failure taxonomy:

- `memorize`: too narrow 120, variable-length inference 120, optional region 80
- `heuristic`: success 214, variable-length inference 49, optional region 21, too_broad 19, character-class inference 11, too narrow 6
- `grex_default`: too narrow 120, variable-length inference 120, optional region 80
- `grex_digits`: variable-length inference 106, success 87, too narrow 79, optional region 47, character-class inference 1
- `grex_words`: character-class inference 125, optional region 100, variable-length inference 56, too broad 20, mixed population 11, success 8
- `grex_repetitions`: too narrow 120, variable-length inference 120, optional region 80
- `grex_combined`: success 153, variable-length inference 92, optional region 45, character-class inference 17, mixed population 10, too broad 3
- `fast`: invalid_output 8

## bench_a

| System | Semantic accuracy | Macro accuracy | Positive recall | Negative rejection | Valid | Catastrophic | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| memorize | 55.1% | 55.1% | 10.2% | 100.0% | 100.0% | 0.0% | 0.01 | 0.01 |
| heuristic | 84.4% | 84.4% | 88.2% | 80.6% | 100.0% | 21.4% | 0.16 | 0.45 |
| grex_default | 55.1% | 55.1% | 10.2% | 100.0% | 100.0% | 0.0% | 1.50 | 2.64 |
| grex_digits | 59.0% | 59.0% | 18.1% | 99.9% | 100.0% | 0.0% | 1.50 | 2.59 |
| grex_words | 84.9% | 84.9% | 89.5% | 80.3% | 100.0% | 0.0% | 1.34 | 1.70 |
| grex_repetitions | 55.2% | 55.2% | 10.3% | 100.0% | 100.0% | 0.0% | 1.60 | 2.98 |
| grex_combined | 86.8% | 86.8% | 88.1% | 85.4% | 100.0% | 0.0% | 1.32 | 1.78 |
| fast | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 3168.47 | 6312.90 |

`fast` returned no expression on 8 of 8 tasks (accuracy on completed runs: None). A zero in the table is a timeout, not a scored regex.

Failure taxonomy:

- `memorize`: variable-length inference 680, optional region 306, too narrow 200, success 12, character-class inference 2
- `heuristic`: success 548, too_broad 257, optional region 212, variable-length inference 136, character-class inference 25, too narrow 11
- `grex_default`: variable-length inference 680, optional region 306, too narrow 200, success 12, character-class inference 2
- `grex_digits`: variable-length inference 634, optional region 286, too narrow 196, success 83, character-class inference 1
- `grex_words`: optional region 418, success 375, variable-length inference 192, character-class inference 123, too broad 55, mixed population 37
- `grex_repetitions`: variable-length inference 679, optional region 307, too narrow 200, success 12, character-class inference 2
- `grex_combined`: success 496, optional region 344, variable-length inference 218, character-class inference 101, too broad 25, mixed population 16
- `fast`: invalid_output 8

## BENCH-D sample efficiency (heuristic)

| Visible examples | Semantic accuracy |
| ---: | ---: |
| 2 | 81.1% |
| 3 | 86.2% |
| 5 | 92.5% |
| 10 | 94.0% |
| 20 | 95.3% |

## Gate 1

On this BENCH-C file the structural heuristic reaches 93.2% micro semantic accuracy and 92.1% macro accuracy across domains. Invoice-number columns are capped so they do not dominate the micro average. grex's combined digit/word/repetition mode is the next symbolic system. Default grex matches the memorization floor. fAST's reference search returned no expression within 3 seconds on the short tasks it was given, so the zero scores are timeouts, not judged regexes. Gate 1 still needs two frontier models on this same file. No API key is configured, so those cells are empty. The decision rule is an 8–10 point gap between the best symbolic accuracy and the frontier models. Do not call the gate from the symbolic column alone. If frontier models also land near the heuristic, the spec says not to train a neural model.
