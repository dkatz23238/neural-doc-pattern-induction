## Factorized candidate prior

The generator, the compositional corpus, the 160-d support vector, the optimizer, the 20,000 steps, and the validation checkpoint rule stay frozen. This pass changes the score. A candidate is a sum of local pieces: field, width, repeat, optional, affix, sequence, and alternation. Each piece has its own projection. The repeat projection does not receive the child, and the field projection does not receive the quantifier. An interaction term is `tanh(beta)` times the cosine of those two hidden states, with the states detached, so the term stays inside [-1, 1] and cannot rotate the pieces into a private code for one pair. Auxiliary targets mark whether the field, the width, the repeat, the optional, and the affix match some probe-perfect candidate. A rejected candidate that got one piece right still trains that piece. Execution stays symbolic. The symbolic weights are untouched. BENCH-C is not a selection metric. BENCH-E stays sealed. `needle()` is unchanged.

The corpus is the compositional one: seed 2507, 50,000 / 10,000 / 20,000 tasks. This machine has no CUDA. A run takes about a minute and a half.

| Parameters | Width | Val top-1 | OOD top-1 |
| ---: | ---: | ---: | ---: |
| 0 |  | 0.315 | 0.390 |
| 13,927 | 48 | 0.847 | 0.513 |
| 41,767 | 144 | 0.850 | 0.516 |

Val and OOD are the mean of seeds 0 and 1 at the best validation checkpoint. The 0-parameter row is the fixed symbolic score. The flat 9,737-parameter prior was 0.888 / 0.296. The AST prior was 0.937 / 0.314 at 15,065 parameters.

Per seed, validation / holdout:

| Parameters | Seed 0 | Seed 1 |
| ---: | ---: | ---: |
| 0 | 0.315 / 0.390 | 0.315 / 0.390 |
| 13,927 | 0.847 / 0.514 | 0.848 / 0.511 |
| 41,767 | 0.847 / 0.504 | 0.852 / 0.528 |

The 13,927-parameter checkpoints are step 16,000. The 41,767-parameter checkpoints are step 12,000 and step 16,000.

Held-out families, seed 0 / seed 1. `repeat_width` is the in-distribution repeat family. The AST column is the 15,065-parameter tree encoder. The flat column is the 9,737-parameter compositional prior.

| Family | Symbolic | Flat 10k | AST 15k | Factor 14k | Factor 42k |
| --- | ---: | ---: | ---: | ---: | ---: |
| repeat_interval | 0.814 | 0.049 / 0.068 | 0.103 / 0.341 | 0.730 / 0.701 | 0.692 / 0.766 |
| prefix_charset_optional | 0.000 | 0.065 / 0.000 | 0.000 / 0.000 | 0.000 / 0.003 | 0.000 / 0.005 |
| lexicon_suffix | 0.357 | 0.763 / 0.829 | 0.796 / 0.643 | 0.812 / 0.831 | 0.821 / 0.813 |
| repeat_width (validation) | 0.240 | 0.360 / 0.392 | 0.999 / 0.994 | 0.969 / 0.964 | 0.969 / 0.965 |

`charset_width`, which the AST prior scored at 0.985 / 0.974, is 0.542 / 0.539 at 14k and 0.541 / 0.539 at 42k. That drop is most of the validation gap. Lexicon, optional prefix, and stem families stay above 0.95.

Renaming invariance on 32 letter-prefix tasks is 1.000 for the candidate features, the frozen support embedding, and the factor scores. Auxiliary accuracy on the best 14k checkpoints is 0.920 validation and 0.712 holdout for seed 0, and 0.919 / 0.749 for seed 1.

| Gate | Measured |
| --- | ---: |
| Validation top-1 >= 0.87 | 0.847 |
| Holdout top-1 > 0.60 | 0.513 |
| `repeat_interval` > 0.50, both seeds | 0.730 / 0.701 |
| `prefix_charset_optional` > 0.50, both seeds | 0.000 / 0.003 |
| `lexicon_suffix` holds the flat 10k scores | 0.812 / 0.831 against 0.763 / 0.829 |
| Renaming invariance | 1.000 |

`repeat_interval` moves from 0.103 / 0.341 on the AST prior to 0.730 / 0.701 at 13,927 parameters, on both seeds, and stays in that range at 41,767. The symbolic score on the same rows is 0.814. `lexicon_suffix` holds the flat prior. `prefix_charset_optional` stays at zero. Mean holdout top-1 is 0.513, above the symbolic 0.390 and the AST 0.314, and below 0.60. The same width does not raise it. Validation is 0.847, below the AST 0.937, with `charset_width` near 0.54.

The additive repeat piece transfers to an integer range it was not trained on. The optional-prefix plus character-class combination does not. That split is the result of this objective. BENCH-E stays sealed.
