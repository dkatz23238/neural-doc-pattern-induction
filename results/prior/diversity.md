## Compositional prior

The candidate generator, the training script, and the model family stay frozen. This pass only changes how latent languages are sampled. A training language is a product of class, width, affix, optionality, repeat, interval, and lexicon factors. Three products are held out of training and validation: optional prefix plus a character class, a repeated field plus a numeric interval, and a lexicon plus a literal suffix. A row is stored only when the frozen expander emits that program. Checkpoints are chosen by validation top-1. The symbolic weights are the same fixed score as the previous curve. BENCH-C is not a selection metric. BENCH-E stays sealed. `needle()` is unchanged.

The corpus is 50,000 training tasks, 10,000 validation tasks, and 20,000 holdout tasks, seed 2507. Generation kept 50,000 of 50,008 training draws (0.9998), 10,000 of 10,001 validation draws, and 20,000 of 20,004 holdout draws. The candidate-pool oracle on the kept rows is 1.000.

| Split | Tasks | Unique latent ASTs | Entropy (bits) | Max AST share | Exact duplicate fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 50,000 | 1,881 | 9.544 | 0.0145 | 0.0217 |
| Validation | 10,000 | 1,444 | 9.407 | 0.0155 | 0.0075 |
| Holdout | 20,000 | 2,858 | 8.111 | 0.0302 | 0.0003 |

The most common training program is 1.45% of the corpus (`SEQ(lower{3,3},COPY(0:3-4),LEX[file_extension])`). The holdout families are absent from the training shards.

| Parameters | Width | Hidden | Val top-1 | OOD top-1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 |  |  | 0.315 | 0.390 |
| 9,737 | 32 | 24 | 0.888 | 0.296 |
| 30,145 | 64 | 64 | 0.888 | 0.279 |
| 105,505 | 192 | 96 | 0.864 | 0.424 |

Val and OOD are the mean of seeds 0 and 1 at the best validation checkpoint. The symbolic score does not use the seed, so both seed columns are that one evaluation. This machine has no CUDA.

Per seed, validation / holdout:

| Parameters | Seed 0 | Seed 1 |
| ---: | ---: | ---: |
| 0 | 0.315 / 0.390 | 0.315 / 0.390 |
| 9,737 | 0.886 / 0.292 | 0.891 / 0.299 |
| 30,145 | 0.889 / 0.297 | 0.888 / 0.261 |
| 105,505 | 0.845 / 0.540 | 0.883 / 0.308 |

Same 9,737-parameter model, 20,000 steps. The 50,000-task row is the mean above.

| Training tasks | Val top-1 | OOD top-1 |
| ---: | ---: | ---: |
| 5,000 | 0.875 | 0.226 |
| 50,000 | 0.888 | 0.296 |

Seed 0 / seed 1 at 5,000 tasks: validation 0.879 / 0.872, holdout 0.185 / 0.267.

Where the 10k model moves, at its best validation checkpoint, seed 0 / seed 1. The symbolic column is the 0-parameter score on this corpus.

| Family | Symbolic | 10k validation | 10k holdout |
| --- | ---: | ---: | ---: |
| charset_width | 0.644 | 0.983 / 0.989 |  |
| interval | 0.211 | 0.867 / 0.856 |  |
| lexicon | 0.779 | 0.947 / 0.945 |  |
| prefix_charset | 0.069 | 0.987 / 0.985 |  |
| prefix_interval | 0.112 | 0.867 / 0.867 |  |
| prefix_lexicon | 0.524 | 0.930 / 0.947 |  |
| prefix_optional | 0.000 | 1.000 / 1.000 |  |
| prefix_width | 0.170 | 0.932 / 0.946 |  |
| repeat_width | 0.240 | 0.360 / 0.392 |  |
| stem_lexicon | 0.635 | 0.996 / 0.994 |  |
| suffix_width | 0.154 | 0.926 / 0.940 |  |
| width | 0.550 | 0.876 / 0.884 |  |
| lexicon_suffix | 0.357 |  | 0.763 / 0.829 |
| prefix_charset_optional | 0.000 |  | 0.065 / 0.000 |
| repeat_interval | 0.814 |  | 0.049 / 0.068 |

`repeat_width` stays near 0.36 at 30k (0.355 / 0.357) and at 100k (0.364 / 0.369). On the holdout, `repeat_interval` is 0.090 / 0.133 at 30k and 0.353 / 0.171 at 100k, against the symbolic score of 0.814. `prefix_charset_optional` is 0.646 on 100k seed 0 and 0.000 on 100k seed 1.

Renaming invariance on 32 letter-prefix tasks is 1.000 for the candidate features, the frozen support embedding, and the MLP scores. Digit endpoints and lexicon tokens were left in place.

| Gate | Measured |
| --- | ---: |
| Validation top-1 > 0.90 | 0.888 |
| Structural OOD top-1 > 0.80 | 0.424 |
| Stretch OOD > 0.85 | 0.424 |
| Candidate oracle | 1.000 |
| Renaming invariance | 1.000 |
| 50k ahead of 5k | validation +0.013, holdout +0.070 |

The compositional corpus has 1,881 distinct training programs and 9.544 bits of signature entropy, and the learned prior still does not clear the holdout. The best mean holdout is 0.424 at 105,505 parameters, below 0.80, and the 9,737-parameter holdout of 0.296 is below the 0-parameter score of 0.390. Fifty thousand tasks raise validation from 0.875 to 0.888. The repeated-field families are the gap: in-distribution `repeat_width` stays near 0.36 at every size, and the held-out repeat-plus-interval program stays below the symbolic score. The next change is the candidate representation. The parameter sweep stops at 100k. BENCH-E stays sealed.
