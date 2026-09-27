# Latent schema

A tiny set encoder reads the visible strings and predicts a short schema: literal spans, character classes, repeated groups, optional tails, and literal alternations. A deterministic compiler turns that schema into a regex. Nothing in the model ranks a candidate pool.

Training tasks are sampled from the schema grammar. The epoch is chosen on a shifted split with held-out delimiters and longer prefixes. BENCH-C is scored once.

## Reading

The schema compiler covers 55.0% of BENCH-C exactly. On the shifted validation split the greedy schema matches the generator regex 0.4% of the time. On BENCH-C the compiled schema scores 7.0% with regressions 95.9%. The heuristic scores 93.2% and the old rule 96.2%.

Literal-prefix recovery on BENCH-C is 51.1%. Repeat recovery is 49.5%, optional recovery is 80.5%, and alternation recovery is 79.4%.

In-distribution exact match at the chosen epoch is 17.2%. The shifted split, whose delimiters are withheld from training, matches 0.4%. Copying a delimiter the grammar did not use in training does not work yet.

When the compiled regex does not match the visible strings, the practical system keeps the heuristic and scores 93.1%. Binary property scores are high where the common label is absence. The executable result is the compiled regex. On tasks the fragment can express, the gold schema scores 100%, so this gap is the encoder.

The epoch was chosen on the shifted split before BENCH-C was scored.

## Protocol

Parameters 291896. Epoch 7 of 8. Shifted-split exact regex match 0.4%. In-distribution exact match 17.2%, logged and not used to stop. Streamed tasks per epoch: 32000.

## C

Untouched document tasks, scored once. The failure buckets below set recall and rejection equal to semantic accuracy, so they are coarser than the buckets in the earlier oracle reports.

| System | Semantic accuracy | Macro | Improved | Regressed |
| --- | --- | --- | --- | --- |
| heuristic | 93.2% | 92.1% | 0.0% | 0.0% |
| old code-length rule | 96.2% | 95.7% | 14.4% | 0.7% |
| compiled schema | 7.0% | 5.4% | 0.7% | 95.9% |
| schema, else heuristic | 93.1% | 92.1% | 0.7% | 3.2% |
| gold schema, expressible tasks only | 100.0% | 100.0% | 0.0% | 0.0% |

Property recovery against the gold regex:

| Property | Accuracy |
| --- | --- |
| prefix | 51.1% |
| suffix | 82.8% |
| alternation | 79.4% |
| optional | 80.5% |
| repeat | 49.5% |
| digit | 93.3% |
| upper | 85.2% |
| lower | 79.4% |
| hex | 86.8% |
| delim | 57.8% |

Heuristic-failure families:

| Failure | n | heuristic | old rule | schema |
| --- | --- | --- | --- | --- |
| success | 417 | 98.8% | 98.9% | 7.6% |
| mixed population | 47 | 62.2% | 89.6% | 1.9% |
| optional region | 44 | 84.7% | 87.6% | 14.3% |
| variable-length inference | 21 | 84.3% | 84.2% | 0.0% |
| character-class inference | 17 | 83.7% | 99.1% | 0.0% |
| semantic class recognition | 7 | 67.6% | 67.6% | 0.0% |
| alternation | 1 | 89.6% | 89.6% | 0.0% |

## D

Sample-size curve. Not used to fit or to stop.

| System | Semantic accuracy | Macro | Improved | Regressed |
| --- | --- | --- | --- | --- |
| heuristic | 89.8% | 89.8% | 0.0% | 0.0% |
| old code-length rule | 93.8% | 93.8% | 16.6% | 0.0% |
| compiled schema | 17.4% | 17.4% | 3.8% | 95.3% |
| schema, else heuristic | 89.5% | 89.5% | 3.8% | 14.7% |
| gold schema, expressible tasks only | 100.0% | 100.0% | 0.0% | 0.0% |
