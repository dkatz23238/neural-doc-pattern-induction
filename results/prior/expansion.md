## Expansion prior

The 1,158-parameter evidence scorer stays frozen. A separate prior sees only the generalization being proposed: the observed character kinds, the proposed superclass, how many unseen symbols it adds, the expansion ratio, the observed and candidate widths, whether that width is open, the integer hull against the candidate window, left and right slack, the expansion ratio, round or conventional endpoints, and, for a repeat, the slack past the observed counts. A hypothesis that is the observed hull adds zero. A hypothesis that does not contain the observations adds zero.

The program score is `evidence_score + λ * latent_prior_score`. λ is a learned positive scale. Local training uses rows where a compatible conservative hypothesis and a compatible expansion disagree about the gold label. The program term is added only when the evidence gap is below 1, which is the most one tanh-bounded node can move at λ = 1. A larger correct evidence gap is held in place. Checkpoint selection is validation top-1 of the sum. The labels are the synthetic generator's probe-perfect programs, so a win here is a win on that generator's expansion conventions.

Parameters 1765. Evidence parameters 1158. λ at the selected checkpoints 0.546 / 0.488. Ambiguous training rows 48420 / 50000.

### Gates

Local accuracy is strict argmax over the compatible pool. Several probe-perfect programs can share a pool, so a conservative row can be gold on the same example as an expansion. The chose-an-expansion rate is the preference. Accuracy can stay high while that rate stays at zero, which means the argmax is a different gold row.

| Gate | Val seed 0 | Val seed 1 | OOD seed 0 | OOD seed 1 |
| --- | ---: | ---: | ---: | ---: |
| Charset expansion | 0.748 (n=7088) | 0.742 (n=7088) | 0.524 (n=16771) | 0.512 (n=16771) |
| Open width | 0.632 (n=3310) | 0.632 (n=3310) | 0.653 (n=6036) | 0.653 (n=6036) |
| Integer-range expansion | 0.451 (n=3315) | 0.438 (n=3315) | 0.280 (n=6664) | 0.278 (n=6664) |
| Loose repeat | 0.000 (n=7) | 0.000 (n=7) | 0.916 (n=2748) | 0.916 (n=2748) |

Chose-an-expansion rate on those same slices. A low rate means the prior left the conservative hypothesis in front.

| Slice | Val seed 0 | Val seed 1 | OOD seed 0 | OOD seed 1 |
| --- | ---: | ---: | ---: | ---: |
| Charset expansion | 0.991 | 0.986 | 1.000 | 1.000 |
| Open width | 0.000 | 0.000 | 0.000 | 0.000 |
| Integer-range expansion | 0.983 | 0.985 | 0.756 | 0.877 |
| Loose repeat | 0.000 | 0.000 | 0.000 | 0.000 |

Where the gold hypothesis is the conservative one and an expansion is also compatible, the prior should keep the conservative row.

| Conservative gold | Val seed 0 | Val seed 1 | OOD seed 0 | OOD seed 1 |
| --- | ---: | ---: | ---: | ---: |
| Exact alphabet | 0.014 (n=70) | 0.014 (n=70) | 0.000 (n=423) | 0.000 (n=423) |
| Exact width | 0.887 (n=3508) | 0.887 (n=3508) | 0.749 (n=7697) | 0.749 (n=7697) |
| Integer hull | 0.889 (n=54) | 0.907 (n=54) | n=0 | n=0 |
| Tight repeat | 1.000 (n=609) | 1.000 (n=609) | 1.000 (n=2634) | 1.000 (n=2634) |

### Whole program

Validation top-1 0.730 / 0.728 against evidence 0.721 / 0.717. OOD top-1 0.474 / 0.447 against evidence 0.492 / 0.469.

Prefix width is the program accuracy on `prefix_optional`, `prefix_width`, `width`, and `suffix_width`. Charset expansion at program level is `prefix_charset_optional`. Integer-range expansion at program level is `interval` and `repeat_interval`.

| Family | Evidence seed 0 | Evidence seed 1 | Combined seed 0 | Combined seed 1 |
| --- | ---: | ---: | ---: | ---: |
| `charset_width` | 0.961 | 0.957 | 0.966 | 0.969 |
| `interval` | 0.483 | 0.488 | 0.480 | 0.469 |
| `lexicon` | 0.917 | 0.913 | 0.920 | 0.926 |
| `prefix_charset` | 0.931 | 0.919 | 0.947 | 0.948 |
| `prefix_interval` | 0.716 | 0.708 | 0.714 | 0.706 |
| `prefix_lexicon` | 0.872 | 0.875 | 0.874 | 0.881 |
| `prefix_optional` | 0.087 | 0.065 | 0.036 | 0.014 |
| `prefix_width` | 0.673 | 0.677 | 0.751 | 0.757 |
| `repeat_width` | 0.947 | 0.941 | 0.956 | 0.971 |
| `stem_lexicon` | 0.980 | 0.979 | 0.985 | 0.976 |
| `suffix_width` | 0.646 | 0.654 | 0.701 | 0.713 |
| `width` | 0.689 | 0.668 | 0.694 | 0.690 |
| `lexicon_suffix` | 0.952 | 0.933 | 0.929 | 0.912 |
| `prefix_charset_optional` | 0.000 | 0.000 | 0.000 | 0.000 |
| `repeat_interval` | 0.526 | 0.473 | 0.494 | 0.430 |

Decisions flipped against the evidence argmax: validation 0.109 / 0.118 (median evidence margin 0.000 / 0.000), OOD 0.140 / 0.182 (median evidence margin 0.132 / 0.060).

Letter-rename invariance 1.000 / 1.000. BENCH-E stays sealed. `needle()` is unchanged.

Charset expansion is a real preference: the prior leaves the observed alphabet on nearly every row, and the chosen superclass is gold on about three quarters of the validation rows and about half of the holdout rows. `prefix_charset_optional` stays at 0. The supports in that family never show a digit, and the superclass the prior learned elsewhere does not name the unseen alphanumeric gold.

Open width is not a preference. The chosen quantifier is open on none of those rows, on either seed. The same is true of a loose repeat: the chosen bound is the observed count. Validation has only 7 loose-repeat rows. On holdout the tight bound is itself gold on 0.916 of them, which is why the accuracy column is high.

`prefix_width` and `suffix_width` rise by widening a bound that stays closed. `prefix_optional` falls. The width head did not repair the optional prefix.

Integer expansion is also a preference, and the chosen window is gold on under half of the validation rows and under a third of the holdout rows. `interval` stays near the evidence score. `repeat_interval` falls, and it remains below the additive scorer's 0.730 / 0.701.

`repeat_width` and `charset_width` stay with the frozen scorer. `lexicon_suffix` drops by about 0.02. Whole-program validation rises by under 0.01. Whole-program holdout falls by about 0.02. λ stays near one half, and the flips sit on evidence ties. The prior learned this generator's charset superclass and a bounded width stretch. It did not learn when a wider integer, an open width, or a loose repeat is the probe-perfect program. That is a property of these labels. The two-layer split is the v1 boundary; a broader corpus is the next measurement.
