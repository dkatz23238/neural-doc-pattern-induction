## Candidate-relative scores

The constituent matcher can read that a run repeats. It cannot choose among `Repeat(2..4)`, `Repeat(1+)`, and `Repeat(exactly 2)`, because the repeat query sees a count and not the candidate's bounds. The same gap separates "this field is numeric" from `IntegerRange(2020, 2035)`.

This scorer gives each node the statistics of the span that node matches. A repeat node sees its own bounds and the observed firing counts. It does not receive the child's type. An integer node sees its own endpoints against the numbers it consumed. A charset node sees its alphabet against the characters it consumed. One network per node type, hidden width 16, 1,158 parameters. The program score is the sum. There is no support embedding and no cross-node term.

The local target is which hypothesis appears in a probe-perfect program. Local node selection is recorded first. Whole-program top-1 is the checkpoint criterion, best validation, two seeds, 20,000 steps, batch 32, CPU. `needle()` is unchanged. BENCH-E stays sealed.

### Local node selection

| Node | Family | What is selected | Seed 0 | Seed 1 |
| --- | --- | --- | ---: | ---: |
| Repeat | `repeat_width` | gold bounds | 0.990 | 0.990 |
| Repeat | `repeat_width` | observed min and max | 1.000 | 1.000 |
| Repeat | `repeat_interval` | gold bounds | 0.965 | 0.965 |
| Repeat | `repeat_interval` | observed min and max | 1.000 | 1.000 |
| Charset | `charset_width` | gold alphabet | 0.937 | 0.930 |
| Charset | `prefix_charset_optional` | gold alphabet | 0.000 | 0.000 |
| Integer | `interval` | gold window | 0.546 | 0.548 |
| Integer | `repeat_interval` | gold window | 0.167 | 0.137 |
| Integer | `repeat_interval` | observed hull | 0.000 | 0.000 |
| Lexicon | `lexicon_suffix` | gold lexicon | 1.000 | 1.000 |
| Optional | `prefix_optional` | gold optional | 1.000 | 1.000 |
| Literal | `repeat_interval` | gold literal | 1.000 | 1.000 |

On `repeat_interval` the chosen integer contains every observed value on 0.786 of seed 0, is one of the gold windows on 0.168, and is the observed hull on 0.000. A hull hypothesis is in the pool on every row. The gold windows are wider than that hull, about 4.7 of 15 hypotheses. The packet carries the slack, the expansion ratio, and a hull bit. The scorer uses containment. It does not pick the hull, and it does not pick the probe-perfect expansion.

The pooled validation repeat-node figure is 0.513 / 0.513. `prefix_charset`, `prefix_lexicon`, and `stem_lexicon` contribute gold open-repeats whose supports do not vary in count, and those rows score 0.000. The two repeat families themselves are the rows above.

### Whole-program top-1

| Gate | Seed 0 | Seed 1 |
| --- | ---: | ---: |
| `repeat_width` | 0.947 | 0.941 |
| `charset_width` | 0.961 | 0.957 |
| `repeat_interval` | 0.526 | 0.473 |
| `prefix_charset_optional` | 0.000 | 0.000 |
| `lexicon_suffix` | 0.952 | 0.933 |
| Validation | 0.721 | 0.717 |
| Holdout | 0.492 | 0.469 |
| Renaming invariance | 1.000 | 1.000 |

`repeat_width` is back above 0.90. The constituent matcher had it at 0.284 / 0.260. `charset_width` stays above 0.95. `repeat_interval` stays below the additive scorer's 0.730 / 0.701. The repeat half of that composition is already ranked at 1.000 tight bounds. The integer half is the miss, so the range packet does not yet carry the program.

`prefix_charset_optional` stays at 0. The supports are uppercase. Charset node selection on that family is 0.000. An alphanumeric class is an unseen digit, which this packet cannot observe.

`prefix_optional` programs are 0.087 / 0.065 while the optional node itself is 1.000 / 1.000. The miss sits on the width attached to the prefix, the same window-ranking gap as the integer head.

Validation is 0.721 / 0.717. In distribution, `interval` is 0.483 / 0.488 and `width` is 0.689 / 0.668. `lexicon_suffix` is 0.952 / 0.933, with the lexicon node at 1.000 / 1.000. Renaming invariance is 1.000 / 1.000.

Repeat bounds and the charset alphabet are now local scores of the span a node consumes. The remaining candidate-parameter gap is which containing integer window is the probe-perfect one. The unseen alphabet on `prefix_charset_optional` is a prior over characters the supports never show.
