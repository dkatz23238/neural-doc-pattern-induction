## Constituent state

The recognizer stays the frozen 1.226M model. The pooled 160-d support vector is not the input. Each probe reads one constituent: the run layout the encoder already computed, or the numeric range of a digit run. Probes train on the train families and are chosen by validation balance. The held-out compositions are `repeat_interval`, `prefix_charset_optional`, and `lexicon_suffix`. The gate is accuracy above 0.90 on both seeds. BENCH-C is not used. BENCH-E stays sealed. `needle()` is unchanged. The machine has no CUDA.

A readout of every run at once is not a readout of one constituent. The joint layout still carries training co-occurrence. Every training interval is a single digit run, and every training suffix is uppercase.

| Readout | Interval on `repeat_interval` | Optional scope on `prefix_charset_optional` | Suffix on `lexicon_suffix` |
| --- | --- | --- | --- |
| Pooled 160-d vector | 0.184 / 0.142 | 0.104 / 0.119 | 0.000 / 0.001 |
| Run layout, all positions | 0.000 / 0.000 | 1.000 / 1.000 | 0.508 / 0.541 |
| Frozen span hidden | 0.052 / 0.007 | 1.000 / 1.000 | 0.000 / 0.000 |
| Class-blind layout | 0.052 / 0.016 | 1.000 / 1.000 | 0.738 / 0.745 |
| Last-run role | 0.000 / 0.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| Digit-run coverage | 1.000 / 1.000 | 1.000 / 1.000 | 0.000 / 0.000 |

The two numbers are seed 0 and seed 1 of a linear probe. Optional scope, the optional bit, repeat, and the character class are already at 1.000 / 1.000 on the unpooled run layout, including the held-out compositions. The pooled vector had optional scope at 0.104 / 0.119. The frozen span hidden keeps optional scope and drops the other two facts.

The suffix is the last run. Training suffixes are uppercase, so a probe of the whole layout uses that class and transfers at 0.508 / 0.541. Dropping class identity raises that to 0.738 / 0.745. Reading only the last run, still without class, transfers at 1.000 / 1.000, and the same probe scores the in-distribution suffix and the held-out optional prefix at 1.000 / 1.000. The role was in the layout. The other runs were the shortcut.

Interval is not in the layout and not in the span hidden. Within-field column constancy does not clear it either: a probe of those channels scores `repeat_interval` at 0.000 / 0.000. On the training shard an interval is one digit run, so the probe treats a second digit run as evidence against the label. The constancy values themselves are only a partial correlate. The fact that does transfer is the numeric coverage of a digit constituent, `(max − min + 1)` divided by the range of that width. One number, four parameters. `repeat_interval` is 1.000 / 1.000. In distribution, interval is 0.964 / 0.964, a repeated digit class is rejected at 1.000 / 1.000, and a spread digit width is rejected at 0.858 / 0.858. That last figure is the soft spot: some width tasks draw a tight handful of integers. The held-out repeated window is tighter than those, so the same threshold catches all of it.

The slices do not substitute for each other. Run presence scores optional scope at 1.000 / 1.000 and interval at 0.000 / 0.000. The repeat flag scores repeat at 1.000 / 1.000 and optional scope at 0.000 / 0.000. The field class scores the held-out character class at 1.000 / 1.000 and the suffix at 0.000 / 0.000. Digit coverage scores the suffix at 0.000 / 0.000.

The three held-out facts are available once each is stored on its constituent. They are not available as one vector, and interval and suffix are not in the frozen span hidden. A scorer that receives the pooled vector, or a concatenation it can use as a co-occurrence code, is still scoring a representation that has dropped the composition.

The matcher does not receive that vector. Each candidate factor queries one constituent: digit coverage for an integer, run presence for an optional, the last-run role for a literal, the repeat flag for a repeat, the field class for a class or a set, and a lexicon-column bit for a lexicon. The interaction is the product of the two projections, because a support term that is shared by every candidate cannot change the ranking. A residual reads the 50-d candidate features and does not see the support. The model has 2,863 parameters. There is one width, two seeds, 20,000 steps, batch 32, on CPU. The checkpoint is the best validation top-1.

| Gate | Seed 0 | Seed 1 |
| --- | ---: | ---: |
| `repeat_interval` | 0.425 | 0.392 |
| `prefix_charset_optional` | 0.000 | 0.000 |
| `lexicon_suffix` | 0.408 | 0.790 |
| `charset_width` | 0.979 | 0.982 |
| Holdout | 0.277 | 0.394 |
| Validation | 0.846 | 0.824 |
| Renaming invariance | 1.000 | 1.000 |

`charset_width` and renaming clear. Nothing else does. `repeat_interval` stays near 0.40 on both seeds, short of 0.70. `prefix_charset_optional` stays at 0. `lexicon_suffix` clears on one seed and not the other. Validation is 0.846 / 0.824, short of 0.87. In distribution, `repeat_width` is 0.284 / 0.260: the repeat query sees a count, not the repeat's bounds, so preservation of the repeat bit is not yet a ranking of repeat programs.

The constituent representation answers the probe question. Facts learned on separate constituents stay readable on the held-out compositions, above 0.90 on both seeds, when each fact is read from its own constituent. The 2,863-parameter matcher does not turn that into the selection gates. A larger scorer on the pooled vector is still the wrong next step. So is Terra, a pretrained backbone, or BENCH-E. The open question is how a candidate node should use a constituent it can already read.
