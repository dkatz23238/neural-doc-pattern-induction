## Support representation audit

The generator, the candidate pool, the frozen 160-d support vector, and the symbolic weights stay as they are. This pass does not train a new prior. It asks whether that vector still contains the facts a compositional prior would need. Each fact has its own probe, a linear map or one hidden layer of width 32, trained on the train families and chosen by validation balance. The held-out compositions are `repeat_interval`, `prefix_charset_optional`, and `lexicon_suffix`. Repeat, interval, optional, optional scope, affix, and lexicon are the family's latent facts. Charset and width are read from the strings. On this corpus every `prefix_charset_optional` support is an uppercase field whose string length changes with the prefix, and every `repeat_interval` support is a digit field whose string length changes with the repeat count. BENCH-C is not used. BENCH-E stays sealed. `needle()` is unchanged. The machine has no CUDA.

A string-side reader, used only to label charset and width and to drive the explicit match, agrees with the family bits: repeat 1.000, optional 0.9999, affix 0.9997 on the training shard, and repeat 1.000, optional 0.997, affix 0.997 on the holdout shard.

| Fact | Parameters | In distribution | Held-out composition |
| --- | ---: | --- | --- |
| Repeat | 322 | `repeat_width` 1.000 / 1.000 | `repeat_interval` 1.000 / 1.000 |
| Interval | 322 | `interval` 1.000 / 1.000, off on `width` 0.996 / 0.996 | `repeat_interval` 0.184 / 0.142 |
| Optional | 322 | `prefix_optional` 1.000 / 1.000, off on `prefix_charset` 1.000 / 1.000 | `prefix_charset_optional` 0.174 / 0.164 |
| Optional scope | 644 | `prefix_optional` 1.000 / 1.000 | `prefix_charset_optional` 0.104 / 0.119 |
| Charset | 644 | `charset_width` 0.994 / 0.992 | `prefix_charset_optional` 0.966 / 0.958 |
| Width | 483 | `repeat_width` 1.000 / 1.000, `prefix_optional` 1.000 / 1.000 | `repeat_interval` 1.000 / 1.000, `prefix_charset_optional` 0.187 / 0.179 |
| Affix | 483 | `suffix_width` 1.000 / 1.000, `prefix_charset` 1.000 / 1.000 | `lexicon_suffix` 0.000 / 0.001, `prefix_charset_optional` 0.475 / 0.693 |
| Lexicon | 322 | `lexicon` 0.948 / 0.980 | `lexicon_suffix` 0.307 / 0.313 |

The two numbers are seed 0 and seed 1 of the linear probe. Validation balance is 0.981 to 1.000 for every fact. The one-hidden-layer probe is about 5,200 parameters. It does not recover the missing bits. Optional scope on `prefix_charset_optional` is 0.000 / 0.000. Interval on `repeat_interval` is 0.000 / 0.269. The lexicon bit on `lexicon_suffix` rises to 0.851 / 0.806, and the suffix role stays at 0.000 / 0.000.

Read together, seed 0 of the linear probes gets both facts of a composition on this fraction of the holdout:

| Composition | Both facts | The pieces |
| --- | ---: | --- |
| `repeat_interval` | 0.184 | repeat 1.000, interval 0.184 |
| `prefix_charset_optional` | 0.104 | optional 0.174, scope 0.104, charset 0.966 |
| `lexicon_suffix` | 0.000 | lexicon 0.307, affix 0.000 |

Repeat is in the vector, including on a repeated integer field. The character class of the held-out optional family is in the vector. The interval bit is in the vector for a bare interval and a prefixed interval, and it is absent once that field is repeated. Optional scope is in the vector for an optional digit prefix, and it is absent once the field is an uppercase class. A suffix is in the vector after a digit field, and it is absent after a lexicon. Those are the compositions the prior could not learn.

The same strings, read before the pool, are then matched to each candidate. Repeat, optional scope, affix, field class, width, and lexicon each add 1 for an agreement and subtract 1 for a contradiction. A set that covers a pure observed class scores 0 on the field piece. The weights are fixed.

| Split | Top-1 | A gold shares the top score |
| --- | ---: | ---: |
| Validation | 0.288 | 0.694 |
| Holdout | 0.160 | 0.553 |
| `repeat_interval` | 0.195 | 0.982 |
| `prefix_charset_optional` | 0.065 | 0.319 |
| `lexicon_suffix` | 0.221 | 0.360 |

On every `prefix_optional` row the chosen candidate ties the gold on repeat, optional, affix, and field, and wins by 2 on width. The latent digit width is open. Every support field has one width, so an exact-width candidate outscores the gold. On `prefix_charset_optional` the winner ties the gold on optional, affix, and width, and wins by 0.681 on the field piece: the supports are uppercase, and a pure uppercase class outscores the alnum set. On `repeat_interval` a gold program shares the winning score on 0.982 of rows. The string facts treat a repeated digit class and a repeated integer range as the same numeric field, so argmax top-1 stays at 0.195.

The pooled vector is where optional scope and interval-under-repeat are lost. Exposing the string facts puts those bits back in front of a scorer, and the fixed match still misses probe-perfect top-1, because an integer window, a character the supports never show, and a width bound no support displays are not among those facts. The next change is the support representation: keep the local facts available, or retrain the encoder so they survive the pool. A larger prior on this vector would be scoring a representation that has already dropped those compositions. Pretrained semantic features become relevant once the structural facts decode and the remaining ambiguity is which closed set a token belongs to. This audit does not justify a GPU run.
