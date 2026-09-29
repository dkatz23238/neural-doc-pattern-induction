## Candidate prior

The candidate generator stays frozen at the 0.980 product-contract oracle. This pass trains only a scorer over candidates that generator already emitted. Each support set is embedded once by the frozen 1.226M encoder (160 dimensions). Each candidate is a vector of 50 structural features: operator counts, bounds, integer-range properties, lexicon identity, and coverage. The objective is listwise cross-entropy. Training loads parquet shards and does no regex synthesis. Device, precision, and parameter count are configuration of one script. This machine has no CUDA, so the curve below is the CPU run of that script.

The corpus is 50,000 training tasks, 10,000 validation tasks, and 20,000 structural holdout tasks, seed 2507. Generation kept 50,000 of 50,008 training draws (0.9998) and every validation and holdout draw. A task is stored only when the frozen expander emits the latent language, so the candidate-pool oracle on the shards is 1.000. Holdout families are unseen intervals, alphabets, lexicons, widths, and a prefix-plus-currency composition.

Checkpoints are chosen by validation top-1. BENCH-C is not a selection metric. BENCH-E stays sealed. `needle()` is unchanged.

| Parameters | Width | Hidden | Val top-1 | OOD top-1 | Score latency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 |  |  | 0.460 | 0.583 | 1.66 ms |
| 9,737 | 32 | 24 | 0.886 | 0.510 | 0.13 ms |
| 30,145 | 64 | 64 | 0.876 | 0.543 | 0.64 ms |
| 105,505 | 192 | 96 | 0.850 | 0.612 | 0.33 ms |
| 289,601 | 320 | 192 | 0.838 | 0.610 | 0.82 ms |

Val and OOD are the mean of seeds 0 and 1 at the best validation checkpoint. The 0-parameter row is a fixed symbolic score, identical across seeds. Latency is one CPU forward of 32 tasks with 32 candidates each.

Per seed, validation / holdout:

| Parameters | Seed 0 | Seed 1 |
| ---: | ---: | ---: |
| 0 | 0.460 / 0.583 | 0.460 / 0.583 |
| 9,737 | 0.883 / 0.498 | 0.889 / 0.522 |
| 30,145 | 0.875 / 0.548 | 0.877 / 0.538 |
| 105,505 | 0.839 / 0.649 | 0.862 / 0.575 |
| 289,601 | 0.835 / 0.630 | 0.841 / 0.589 |

The 9,737-parameter model is the knee. It learns the training families. Further parameters lower validation top-1. Holdout top-1 rises from 0.510 at 10k to 0.612 at 100k and stays there at 300k.

Where the 10k model moves, seed 0 / seed 1:

| Family | Symbolic | 10k validation | 10k holdout |
| --- | ---: | ---: | ---: |
| charset | 0.887 | 0.980 / 0.984 | 0.013 / 0.325 |
| interval | 0.215 | 0.715 / 0.713 | 0.037 / 0.082 |
| lexicon | 0.991 | 0.996 / 0.992 | 0.885 / 0.903 |
| prefix / composition | 0.206 | 0.753 / 0.762 | 1.000 / 0.972 |
| width | 0.004 | 0.973 / 0.994 | 0.557 / 0.330 |

The symbolic holdout column is charset 0.621, interval 0.391, lexicon 0.902, composition 1.000, width 0.000. The learned model picks up width, prefix, and in-distribution intervals. Held-out intervals stay near zero. Held-out alphabets and widths move with the seed.

Same 9,737-parameter model, 20,000 steps, smaller prefixes of the same 50,000 tasks:

| Training tasks | Val top-1 | OOD top-1 |
| ---: | ---: | ---: |
| 5,000 | 0.879 | 0.496 |
| 15,000 | 0.887 | 0.278 |
| 25,000 | 0.886 | 0.509 |
| 50,000 | 0.886 | 0.510 |

Validation accuracy is already at the 50,000-task level with 5,000 draws. The generator repeats a small menu of latent languages, so more copies of that menu do not add conventions.

Renaming invariance on 32 letter-prefix tasks is 1.000 for the candidate features, the frozen support embedding, and the MLP scores. Digit endpoints and lexicon tokens are semantic and were left in place.

| Gate | Measured |
| --- | ---: |
| Synthetic ambiguous top-1 > 0.90 | 0.889 |
| Synthetic candidate-pool oracle > 0.99 | 1.000 |
| OOD ambiguity > 0.85 | 0.612 |
| Renaming invariance | 1.000 |

CPU Gate A is open: in-distribution top-1 is 0.889 and holdout top-1 is 0.612. CPU Gate B: validation saturates at about 10k parameters, and this corpus saturates at about 5k tasks. The GPU gate is closed from this curve. A 1M–5M run is not scheduled. The recognizer stays at 1,226,128 parameters. BENCH-E stays sealed.
