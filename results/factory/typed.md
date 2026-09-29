## Typed scales

The heads and features stay the expansion prior's. Each ambiguity has its own softplus scale, initialized at 0.01. The program score is evidence plus the sum of scale times that head. A scale can shrink toward 0, and then that head leaves the ranking with the frozen evidence scorer.

Training batches stay 16 synthetic ambiguous rows and 16 factory ambiguous rows. The loss, margin, hinge, optimizer, and 20,000-step budget are unchanged. The checkpoint maximizes the mean of synthetic validation top-1 and factory validation top-1. Factory held-out is scored after that choice.

Parameters 1768. Scales at the selected checkpoints, charset / width / integer / repeat: 0.020 / 0.458 / 1.133 / 2.106 and 0.291 / 0.477 / 1.316 / 3.227.

Selected factory validation top-1 0.569 / 0.552 against evidence 0.526 / 0.526. Synthetic validation 0.719 / 0.717 against evidence 0.721 / 0.717.

The gate requires both seeds. On factory held-out, program accuracy on each ambiguity slice is at least the evidence accuracy. Overall held-out accuracy, every synthetic family, and every synthetic ambiguity slice stay within 0.02 of evidence. Retention is separate: integer and loose-repeat held-out programs have to beat evidence.

Seed 0 misses the gate and does not retain integer and loose repeat. Seed 1 misses the gate and retains integer and loose repeat.

### Factory held-out programs

The slice numbers are program top-1 on the rows in that gate, combined against evidence on the same rows.

| Slice | Seed 0 | Seed 1 |
| --- | ---: | ---: |
| All held-out | 0.451 vs 0.472 (n=144) | 0.458 vs 0.458 (n=144) |
| `charset_expansion` | 0.250 vs 0.250 (n=12) | 0.250 vs 0.250 (n=12) |
| `open_width` | 0.462 vs 0.462 (n=13) | 0.462 vs 0.462 (n=13) |
| `integer_expansion` | 0.556 vs 0.500 (n=54) | 0.556 vs 0.463 (n=54) |
| `loose_repeat` | 0.000 vs 0.000 (n=10) | 0.100 vs 0.000 (n=10) |

### Synthetic programs on the same slices

A slice more than 0.02 under evidence is a miss.

| Slice | Val seed 0 | Val seed 1 | OOD seed 0 | OOD seed 1 |
| --- | ---: | ---: | ---: | ---: |
| `charset_expansion` | 0.798 vs 0.799 (n=7088) | 0.799 vs 0.802 (n=7088) | 0.466 vs 0.483 (n=16771) | 0.453 vs 0.465 (n=16771) |
| `open_width` | 0.757 vs 0.760 (n=3310) | 0.759 vs 0.768 (n=3310) | 0.931 vs 0.951 (n=6036) | 0.913 vs 0.932 (n=6036) |
| `integer_expansion` | 0.643 vs 0.658 (n=3315) | 0.640 vs 0.642 (n=3315) | 0.480 vs 0.526 (n=6664) | 0.450 vs 0.473 (n=6664) |
| `loose_repeat` | 0.000 vs 0.000 (n=7) | 0.000 vs 0.000 (n=7) | 0.318 vs 0.282 (n=2748) | 0.317 vs 0.261 (n=2748) |

### Local gates

Local accuracy is the head's own argmax. It does not pass through the scale. Chose-rate is how often that argmax is an expansion, or an open width.

| Gate | Synthetic val seed 0 | seed 1 | Synthetic OOD seed 0 | seed 1 | Factory held-out seed 0 | seed 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `charset_expansion` | 0.741 (n=7088) | 0.743 (n=7088) | 0.529 (n=16771) | 0.528 (n=16771) | 0.917 (n=12) | 0.917 (n=12) |
| `open_width` | 0.632 (n=3310) | 0.632 (n=3310) | 0.653 (n=6036) | 0.653 (n=6036) | 0.462 (n=13) | 0.615 (n=13) |
| `integer_expansion` | 0.146 (n=3315) | 0.157 (n=3315) | 0.276 (n=6664) | 0.269 (n=6664) | 0.574 (n=54) | 0.556 (n=54) |
| `loose_repeat` | 0.000 (n=7) | 0.000 (n=7) | 0.110 (n=2748) | 0.110 (n=2748) | 0.600 (n=10) | 0.600 (n=10) |

| Chose | Synthetic val seed 0 | seed 1 | Synthetic OOD seed 0 | seed 1 | Factory held-out seed 0 | seed 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `charset_expansion` | 0.982 | 0.985 | 1.000 | 1.000 | 1.000 | 1.000 |
| `open_width` | 0.000 | 0.000 | 0.000 | 0.000 | 0.462 | 0.615 |
| `integer_expansion` | 0.985 | 0.980 | 0.998 | 0.998 | 1.000 | 1.000 |
| `loose_repeat` | 0.000 | 0.000 | 0.852 | 0.852 | 0.600 | 0.600 |

### Synthetic families

Combined minus evidence. A family more than 0.02 under evidence is a miss.

Seed 0

| Family | Split | Evidence | Combined | Delta |
| --- | --- | ---: | ---: | ---: |
| `charset_width` | val | 0.961 | 0.964 | 0.003 |
| `interval` | val | 0.483 | 0.463 | -0.020 |
| `lexicon` | val | 0.917 | 0.919 | 0.001 |
| `prefix_charset` | val | 0.931 | 0.909 | -0.022 |
| `prefix_interval` | val | 0.716 | 0.700 | -0.017 |
| `prefix_lexicon` | val | 0.872 | 0.851 | -0.021 |
| `prefix_optional` | val | 0.087 | 0.055 | -0.032 |
| `prefix_width` | val | 0.673 | 0.726 | 0.053 |
| `repeat_width` | val | 0.947 | 0.948 | 0.001 |
| `stem_lexicon` | val | 0.980 | 0.979 | -0.001 |
| `suffix_width` | val | 0.646 | 0.669 | 0.022 |
| `width` | val | 0.689 | 0.725 | 0.036 |
| `lexicon_suffix` | ood | 0.952 | 0.934 | -0.018 |
| `prefix_charset_optional` | ood | 0.000 | 0.000 | 0.000 |
| `repeat_interval` | ood | 0.526 | 0.480 | -0.045 |

Seed 1

| Family | Split | Evidence | Combined | Delta |
| --- | --- | ---: | ---: | ---: |
| `charset_width` | val | 0.957 | 0.964 | 0.007 |
| `interval` | val | 0.488 | 0.473 | -0.015 |
| `lexicon` | val | 0.913 | 0.926 | 0.013 |
| `prefix_charset` | val | 0.919 | 0.919 | 0.000 |
| `prefix_interval` | val | 0.708 | 0.695 | -0.013 |
| `prefix_lexicon` | val | 0.875 | 0.861 | -0.014 |
| `prefix_optional` | val | 0.065 | 0.065 | 0.000 |
| `prefix_width` | val | 0.677 | 0.705 | 0.028 |
| `repeat_width` | val | 0.941 | 0.941 | 0.000 |
| `stem_lexicon` | val | 0.979 | 0.975 | -0.004 |
| `suffix_width` | val | 0.654 | 0.659 | 0.004 |
| `width` | val | 0.668 | 0.696 | 0.028 |
| `lexicon_suffix` | ood | 0.933 | 0.915 | -0.017 |
| `prefix_charset_optional` | ood | 0.000 | 0.000 | 0.000 |
| `repeat_interval` | ood | 0.473 | 0.450 | -0.023 |

### Gate

The gate is missed.

Misses:

- seed 0 `prefix_charset` val 0.909 vs evidence 0.931
- seed 0 `prefix_lexicon` val 0.851 vs evidence 0.872
- seed 0 `prefix_optional` val 0.055 vs evidence 0.087
- seed 0 `repeat_interval` ood 0.480 vs evidence 0.526
- seed 0 ood `integer_expansion` 0.480 vs evidence 0.526 (n=6664)
- seed 0 held-out 0.451 vs evidence 0.472
- seed 1 `repeat_interval` ood 0.450 vs evidence 0.473
- seed 1 ood `integer_expansion` 0.450 vs evidence 0.473 (n=6664)

Letter-rename invariance 1.000 / 1.000.

### Reading

The gate is missed. Separate scales do keep some of the factory signal, and they stop the open-width collapse from the shared scale.

On factory held-out, integer programs beat evidence on both seeds, 0.556 against 0.500 / 0.463 (n=54). Loose-repeat programs beat evidence on seed 1 by one row, 0.100 against 0.000, and tie evidence at 0.000 on seed 0 (n=10). Charset and open-width programs on those held-out rows tie evidence. Overall held-out is 0.451 / 0.458 against evidence 0.472 / 0.458. Seed 1 matches evidence. Seed 0 is 0.021 under, which is past the 0.02 bar.

The open-width head no longer drags the synthetic validation slice. Local open-width accuracy stays 0.632 / 0.632, where the shared scale had fallen to 0.525 / 0.404, and the chose-open rate on the synthetic slices stays 0. Charset scale stays small on seed 0 (0.020) and moderate on seed 1 (0.291). Width scale settles near 0.46 / 0.48. Integer scale settles near 1.13 / 1.32. Repeat scale grows to 2.11 / 3.23.

The integer head still learns a window the synthetic gold does not use. Local integer accuracy on synthetic validation is 0.146 / 0.157. Program accuracy on that validation slice stays within 0.02 of evidence. On synthetic OOD the same slice falls to 0.480 / 0.450 against evidence 0.526 / 0.473, and `repeat_interval` falls with it. Seed 0 also drops `prefix_charset`, `prefix_lexicon`, and `prefix_optional` by just over 0.02. `repeat_width` and `charset_width` stay with evidence.

Factory integer programs can be kept ahead of evidence without the shared scale's open-width damage. The repeat preference shows up in the head (held-out local accuracy 0.600) and barely in the chosen program. Synthetic OOD integer programs still move by more than 0.02. That is short of a v1 where every other family stays put, so the teacher bank stays at this size.

BENCH-E stays sealed. `needle()` is unchanged. The shared-scale checkpoints are not retrained.
