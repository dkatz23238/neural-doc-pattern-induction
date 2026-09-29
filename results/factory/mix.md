## Factory mixture

The evidence scorer stays the frozen 1,158-parameter model. The prior stays the 1,765-parameter expansion prior: same heads, same λ initialization, same evidence margin of 1, same hinge, same AdamW settings, same 20,000 steps, same batch of 32. Each batch is 16 rows from the synthetic ambiguous pool and 16 rows from the factory ambiguous pool. The checkpoint is the one with the highest synthetic validation top-1.

A factory row is a support configuration of one accepted template. The template program and any neighbor that also accepts those supports form the pool. Gold programs are the ones that agree with the template on its positives, near misses, and corner strings. Families come from `data/factory/manifests/split_v1.json` and are not split.

Factory rows 598 train / 116 validation / 144 held out. Ambiguous rows 437 / 97 / 118. Templates 212 / 47 / 46. Training slices: charset expansion 38, open width 50, integer expansion 171, loose repeat 50.

Parameters 1765. Evidence parameters 1158. λ at the mixture checkpoints 1.071 / 1.143. Synthetic-only λ 0.546 / 0.488.

### Gates

Local accuracy is strict argmax over the compatible pool. The chose-an-expansion rate is the preference.

| Gate | Split | Synthetic-only seed 0 | Synthetic-only seed 1 | Mixture seed 0 | Mixture seed 1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Charset expansion | val | 0.748 (n=7088) | 0.742 (n=7088) | 0.742 (n=7088) | 0.743 (n=7088) |
| Charset expansion | ood | 0.524 (n=16771) | 0.512 (n=16771) | 0.494 (n=16771) | 0.494 (n=16771) |
| Open width | val | 0.632 (n=3310) | 0.632 (n=3310) | 0.525 (n=3310) | 0.404 (n=3310) |
| Open width | ood | 0.653 (n=6036) | 0.653 (n=6036) | 0.653 (n=6036) | 0.659 (n=6036) |
| Integer-range expansion | val | 0.451 (n=3315) | 0.438 (n=3315) | 0.149 (n=3315) | 0.148 (n=3315) |
| Integer-range expansion | ood | 0.280 (n=6664) | 0.278 (n=6664) | 0.276 (n=6664) | 0.262 (n=6664) |
| Loose repeat | val | 0.000 (n=7) | 0.000 (n=7) | 0.000 (n=7) | 0.000 (n=7) |
| Loose repeat | ood | 0.916 (n=2748) | 0.916 (n=2748) | 0.110 (n=2748) | 0.750 (n=2748) |

| Slice | Split | Synthetic-only seed 0 | Synthetic-only seed 1 | Mixture seed 0 | Mixture seed 1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Chose charset expansion | val | 0.991 | 0.986 | 0.985 | 0.987 |
| Chose charset expansion | ood | 1.000 | 1.000 | 1.000 | 1.000 |
| Chose open width | val | 0.000 | 0.000 | 0.000 | 0.000 |
| Chose open width | ood | 0.000 | 0.000 | 0.000 | 0.000 |
| Chose integer expansion | val | 0.983 | 0.985 | 0.988 | 0.987 |
| Chose integer expansion | ood | 0.756 | 0.877 | 0.998 | 0.998 |
| Chose loose repeat | val | 0.000 | 0.000 | 0.000 | 0.000 |
| Chose loose repeat | ood | 0.000 | 0.000 | 0.852 | 0.175 |

### Whole program on the synthetic sets

Validation top-1, synthetic-only 0.730 / 0.728 against evidence 0.721 / 0.717. Mixture 0.736 / 0.743 against evidence 0.721 / 0.717.

OOD top-1, synthetic-only 0.474 / 0.447 against evidence 0.492 / 0.469. Mixture 0.469 / 0.456 against evidence 0.492 / 0.469.

| Family | Split | Synthetic-only seed 0 evidence | combined | seed 1 evidence | combined | Mixture seed 0 evidence | combined | seed 1 evidence | combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `charset_width` | val | 0.961 | 0.966 | 0.957 | 0.969 | 0.961 | 0.969 | 0.957 | 0.969 |
| `interval` | val | 0.483 | 0.480 | 0.488 | 0.469 | 0.483 | 0.473 | 0.488 | 0.492 |
| `lexicon` | val | 0.917 | 0.920 | 0.913 | 0.926 | 0.917 | 0.917 | 0.913 | 0.926 |
| `prefix_charset` | val | 0.931 | 0.947 | 0.919 | 0.948 | 0.931 | 0.940 | 0.919 | 0.951 |
| `prefix_interval` | val | 0.716 | 0.714 | 0.708 | 0.706 | 0.716 | 0.698 | 0.708 | 0.659 |
| `prefix_lexicon` | val | 0.872 | 0.874 | 0.875 | 0.881 | 0.872 | 0.869 | 0.875 | 0.878 |
| `prefix_optional` | val | 0.087 | 0.036 | 0.065 | 0.014 | 0.087 | 0.015 | 0.065 | 0.003 |
| `prefix_width` | val | 0.673 | 0.751 | 0.677 | 0.757 | 0.673 | 0.767 | 0.677 | 0.838 |
| `repeat_width` | val | 0.947 | 0.956 | 0.941 | 0.971 | 0.947 | 0.976 | 0.941 | 0.972 |
| `stem_lexicon` | val | 0.980 | 0.985 | 0.979 | 0.976 | 0.980 | 0.948 | 0.979 | 0.926 |
| `suffix_width` | val | 0.646 | 0.701 | 0.654 | 0.713 | 0.646 | 0.719 | 0.654 | 0.798 |
| `width` | val | 0.689 | 0.694 | 0.668 | 0.690 | 0.689 | 0.835 | 0.668 | 0.838 |
| `lexicon_suffix` | ood | 0.952 | 0.929 | 0.933 | 0.912 | 0.952 | 0.908 | 0.933 | 0.911 |
| `prefix_charset_optional` | ood | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `repeat_interval` | ood | 0.526 | 0.494 | 0.473 | 0.430 | 0.526 | 0.499 | 0.473 | 0.457 |

### Held-out templates

Probe-perfect top-1 counts a neighbor that agrees with the template on the probe. Exact template is candidate 0, the template program itself. Factory validation is reported and was not used to pick the checkpoint.

| Measure | Synthetic-only seed 0 | Synthetic-only seed 1 | Mixture seed 0 | Mixture seed 1 |
| --- | ---: | ---: | ---: | ---: |
| Probe-perfect top-1 | 0.431 (n=144) | 0.417 (n=144) | 0.417 (n=144) | 0.438 (n=144) |
| Evidence top-1 | 0.472 (n=144) | 0.458 (n=144) | 0.472 (n=144) | 0.458 (n=144) |
| Exact template | 0.431 (n=144) | 0.417 (n=144) | 0.417 (n=144) | 0.438 (n=144) |
| Evidence exact template | 0.472 (n=144) | 0.458 (n=144) | 0.472 (n=144) | 0.458 (n=144) |
| Charset expansion | 0.917 (n=12) | 0.917 (n=12) | 0.917 (n=12) | 0.917 (n=12) |
| Chose charset expansion | 1.000 | 1.000 | 1.000 | 1.000 |
| Open width | 0.462 (n=13) | 0.462 (n=13) | 0.462 (n=13) | 0.538 (n=13) |
| Chose open width | 0.462 | 0.462 | 0.462 | 0.538 |
| Integer-range expansion | 0.537 (n=54) | 0.537 (n=54) | 0.593 (n=54) | 0.611 (n=54) |
| Chose integer-range expansion | 0.944 | 0.944 | 1.000 | 1.000 |
| Loose repeat | 0.000 (n=10) | 0.000 (n=10) | 0.600 (n=10) | 0.600 (n=10) |
| Chose loose repeat | 0.300 | 0.300 | 0.600 | 0.600 |

Letter-rename invariance, mixture 1.000 / 1.000. Synthetic-only 1.000 / 1.000.

BENCH-E stays sealed. `needle()` is unchanged. The synthetic-only checkpoints in `results/prior/runs/expansion_prior` are the comparison point and were not retrained.

### Reading

Held-out template top-1 stays with the frozen evidence scorer. On 144 held-out rows the mixture scores 0.417 / 0.438, the synthetic-only prior scores 0.431 / 0.417, and evidence scores 0.472 / 0.458. Factory validation, unused for checkpoint selection, moves from 0.552 / 0.500 to 0.483 / 0.474, with evidence at 0.526. The flips sit on evidence ties: the median evidence margin of a flip is about 0.

Charset expansion was already this prior's preference, and it stays there. Validation accuracy is 0.748 / 0.742 for the synthetic-only run and 0.742 / 0.743 for the mixture, with a chose-rate near 0.99. The 12 held-out factory charset rows score 0.917 for both models, chose-rate 1. Open width on the synthetic slices still has chose-rate 0. Validation accuracy on that slice falls from 0.632 to 0.525 / 0.404. On 13 held-out factory rows the chose-open rate is 0.462 for the synthetic-only prior and 0.462 / 0.538 for the mixture.

Integer expansion gets more common and less specific. Synthetic holdout chose-rate moves from 0.756 / 0.877 to 0.998 / 0.998, while validation accuracy of the gold window falls from 0.451 / 0.438 to 0.149 / 0.148. On 54 held-out factory integer rows accuracy moves from 0.537 to 0.593 / 0.611. Loose repeat is the factory slice that changes direction: on 10 held-out rows accuracy moves from 0.000 to 0.600 and the chose-rate moves from 0.300 to 0.600. On the synthetic holdout, where the tight bound is itself gold, accuracy moves from 0.916 to 0.110 / 0.750. The two seeds disagree on that slice.

`repeat_width` and `charset_width` stay with the frozen scorer. `width` rises from 0.694 / 0.690 to 0.835 / 0.838, and `prefix_width` and `suffix_width` rise with it. `stem_lexicon` falls from 0.985 / 0.976 to 0.948 / 0.926. `prefix_optional` falls further, to 0.015 / 0.003. `prefix_charset_optional` stays at 0. Synthetic validation top-1 moves from 0.730 / 0.728 to 0.736 / 0.743. Synthetic holdout stays under the evidence score, 0.469 / 0.456 against 0.492 / 0.469. λ moves from about one half to 1.071 / 1.143.

Those factory gradients come from 212 train templates and 437 ambiguous support sets. The gate slices inside that pool are 38 charset, 50 open width, 171 integer, and 50 loose repeat. At this bank the four ambiguity decisions do not move together, and a held-out template is still chosen by the evidence scorer. The bank is the 300–500 template measurement. A flat result at 1,000–2,000 templates would be the point to treat the representation as the open question.
