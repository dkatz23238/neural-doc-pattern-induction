## AST candidate prior

The generator, the compositional corpus, the 160-d support vector, the loss, the optimizer, the 20,000 steps, and the validation checkpoint rule stay frozen. This pass changes the candidate representation. A program is a tree. A literal, a character class, an integer range, and a lexicon each have a learned leaf. Optional, repeat, sequence, and alternation each have one composition function, reused at every node of that operator. The scorer reads the 160-d support vector, a 32-d or 64-d tree vector, and eight relation features. Execution stays in the symbolic interpreter. Checkpoints are the best validation top-1. The symbolic weights are untouched. BENCH-C is not a selection metric. BENCH-E stays sealed. `needle()` is unchanged.

The corpus is the compositional one: seed 2507, 50,000 / 10,000 / 20,000 tasks, candidate oracle 1.000, 1,881 training programs, 9.544 bits. This machine has no CUDA. A run takes about five to six minutes.

| Parameters | Width | Hidden | Tree | Val top-1 | OOD top-1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 |  |  |  | 0.315 | 0.390 |
| 15,065 | 32 | 24 | 64 | 0.937 | 0.314 |
| 39,889 | 80 | 64 | 32 | 0.935 | 0.182 |
| 90,609 | 112 | 128 | 64 | 0.916 | 0.187 |

Val and OOD are the mean of seeds 0 and 1 at the best validation checkpoint. The 0-parameter row is the fixed symbolic score on this corpus. The flat compositional prior, for scale, was 0.888 / 0.296 at 9,737 parameters and 0.864 / 0.424 at 105,505.

Per seed, validation / holdout:

| Parameters | Seed 0 | Seed 1 |
| ---: | ---: | ---: |
| 0 | 0.315 / 0.390 | 0.315 / 0.390 |
| 15,065 | 0.937 / 0.300 | 0.938 / 0.328 |
| 39,889 | 0.935 / 0.178 | 0.935 / 0.186 |
| 90,609 | 0.910 / 0.146 | 0.922 / 0.227 |

The 39,889-parameter seed 0 checkpoint is step 18,000. Every other learned checkpoint is step 20,000.

Held-out families, seed 0 / seed 1, at that checkpoint. `repeat_width` is the in-distribution repeat family. The flat column is the 9,737-parameter compositional prior.

| Family | Symbolic | Flat 10k | AST 15k | AST 40k | AST 91k |
| --- | ---: | ---: | ---: | ---: | ---: |
| repeat_interval | 0.814 | 0.049 / 0.068 | 0.103 / 0.341 | 0.256 / 0.168 | 0.048 / 0.046 |
| prefix_charset_optional | 0.000 | 0.065 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| lexicon_suffix | 0.357 | 0.763 / 0.829 | 0.796 / 0.643 | 0.279 / 0.391 | 0.390 / 0.637 |
| repeat_width (validation) | 0.240 | 0.360 / 0.392 | 0.999 / 0.994 | 1.000 / 0.985 | 0.999 / 0.996 |

Every probe-perfect candidate set for `repeat_interval` contains both a repeat and an integer range (6,664 rows). Every probe-perfect set for `prefix_charset_optional` contains an optional and a character class (6,669 rows). On `lexicon_suffix`, 1,267 of 6,667 rows include a lexicon node among the probe-perfect candidates; the other gold programs are classes or sets that agree with the probe.

Renaming invariance on 32 letter-prefix tasks is 1.000 for the candidate features, the frozen support embedding, and the AST scores. Digit endpoints and lexicon tokens were left in place. The 15k seed 0 process recorded the score check before it included this encoder; reloading that checkpoint scores 1.000 on the same 32 tasks.

| Gate | Measured |
| --- | ---: |
| Validation top-1 >= 0.87 | 0.937 |
| Holdout top-1 > 0.60 | 0.314 |
| `repeat_interval` > 0.50 | 0.341 on the better seed |
| `prefix_charset_optional` > 0.50, both seeds | 0.000 / 0.000 |
| `lexicon_suffix` holds the flat 10k scores | 0.796 / 0.643 against 0.763 / 0.829 |
| Renaming invariance | 1.000 |

The shared repeat function learns the repeats that occur in training: validation `repeat_width` moves from about 0.36 on the flat prior to 0.99. The held-out combination does not follow it. `repeat_interval` rises on some seeds and stays below 0.50, then the 90,609-parameter model returns to the flat prior's range, against a symbolic score of 0.814. `prefix_charset_optional` is 0.000 at every size and both seeds. `lexicon_suffix` stays near the flat prior on one 15k seed and drops on the other seeds and the larger models. Mean holdout top-1 is 0.314 at 15,065 parameters, under the symbolic 0.390 and under 0.60, and larger trees score the holdout lower.

The generator still offers a perfect candidate on the kept rows, and the training corpus is the same 1,881-program distribution. Extra parameters in this encoder do not raise the holdout. The training-set size was left at 50,000. The tree was the only change, and it removes the in-distribution repeat failure while the three held-out operator combinations stay below the diagnostic gates. That does not support the claim that a shared recursive encoder is enough for those compositions. The 0.85 holdout target is untouched. BENCH-E stays sealed.
