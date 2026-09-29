## Typed relation prior

The generator, the compositional corpus, the 160-d support vector, the optimizer, the 20,000 steps, and the validation checkpoint rule stay frozen. The additive pieces stay too: field, width, repeat, optional, affix, sequence, and alternation. The repeat projection still does not see its child. This pass adds three relations and nothing else. Field pairs with width through a rank-8 product of those two hidden states. An optional pairs with the kind of constituent it governs. An affix pairs with the kind of field it attaches to, including whether it sits as a prefix, a suffix, or an infix. Scope embeddings start at zero, so an attachment absent from training adds nothing. Execution stays symbolic. The symbolic weights are untouched. BENCH-C is not a selection metric. BENCH-E stays sealed. `needle()` is unchanged.

The corpus is the compositional one: seed 2507, 50,000 / 10,000 / 20,000 tasks. This machine has no CUDA. The 40k and 100k widths were not trained. The 15k model missed the compositional gates, so there is no parameter curve and no data curve.

| Parameters | Width | Rank | Val top-1 | OOD top-1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 |  |  | 0.315 | 0.390 |
| 15,757 | 48 | 8 | 0.848 | 0.290 |

Val and OOD are the mean of seeds 0 and 1 at the best validation checkpoint, both step 16,000. The 0-parameter row is the fixed symbolic score. The additive prior at 13,927 parameters was 0.847 / 0.513.

Per seed, validation / holdout:

| Parameters | Seed 0 | Seed 1 |
| ---: | ---: | ---: |
| 0 | 0.315 / 0.390 | 0.315 / 0.390 |
| 15,757 | 0.847 / 0.317 | 0.848 / 0.264 |

The additive column is the 13,927-parameter factor prior.

| Family | Symbolic | Additive 14k | Relations 16k |
| --- | ---: | ---: | ---: |
| repeat_interval | 0.814 | 0.730 / 0.701 | 0.426 / 0.324 |
| prefix_charset_optional | 0.000 | 0.000 / 0.003 | 0.000 / 0.000 |
| lexicon_suffix | 0.357 | 0.812 / 0.831 | 0.525 / 0.467 |
| charset_width (validation) | 0.644 | 0.542 / 0.539 | 0.536 / 0.534 |
| repeat_width (validation) | 0.240 | 0.969 / 0.964 | 0.969 / 0.965 |

Renaming invariance on 32 letter-prefix tasks is 1.000 for the features, the frozen support embedding, and the relation scores.

| Gate | Measured |
| --- | ---: |
| Validation top-1 >= 0.87 | 0.848 |
| Holdout top-1 > 0.60 | 0.290 |
| `repeat_interval` >= 0.65, both seeds | 0.426 / 0.324 |
| `prefix_charset_optional` > 0.50, both seeds | 0.000 / 0.000 |
| `charset_width` > 0.80 | 0.536 / 0.534 |
| `lexicon_suffix` holds the additive scores | 0.525 / 0.467 against 0.812 / 0.831 |
| Renaming invariance | 1.000 |

In-distribution `repeat_width` stays at 0.969 / 0.965. Held-out `repeat_interval` falls from 0.730 / 0.701 to 0.426 / 0.324. `prefix_charset_optional` stays at zero. `charset_width` stays near 0.54. `lexicon_suffix` falls from 0.812 / 0.831 to 0.525 / 0.467. Mean holdout top-1 is 0.290, below the additive 0.513 and below the symbolic 0.390.

The three relations do not restore the missing compositions, and they cost the repeat transfer that the additive score had learned. The 40k model, the 100k model, and a larger corpus were not run. BENCH-E stays sealed.
