# Version-space chart

P2a freezes the L1 recognizer and replaces the scored beam with a bottom-up version space. Every gold local fact is present in every condition. Extra hypotheses are added on purpose: one per head, two per head, then the frozen model's own top 8. A partial program is dropped only when it misses a support. Distinct repeat bounds stay in the set together.

The program oracle is the fraction of tasks whose version space contains a support-consistent program that matches the hidden probe. The beam column is the old chart, limited to eight programs, on the same facts. BENCH-C is not scored.

Draws per compose family: 16. Draws per D1 schema: 4. Elapsed 93s.

| Condition | Compose oracle | Compose beam@8 | Compose programs | Truncated | D1 oracle | D1 beam@8 |
| --- | --- | --- | --- | --- | --- | --- |
| Gold facts | 1.000 | 1.000 | 1.0 | 0 | 1.000 | 0.906 |
| Gold + 1 distractor/head | 1.000 | 0.944 | 1.4 | 0 | 1.000 | 0.844 |
| Gold + 2 distractors/head | 1.000 | 0.806 | 1.9 | 0 | 1.000 | 0.781 |
| Gold ∪ model top 8 | 1.000 | 0.632 | 173.7 | 0 | 1.000 | 0.781 |

## Gold facts union the model's top 8

| Family | Version-space oracle | Beam@8 oracle | Open-repeat oracle | Programs | Truncated |
| --- | --- | --- | --- | --- | --- |
| primitive | 1.000 | 1.000 | 0.000 | 334.0 | 0 |
| repeat | 1.000 | 0.250 | 1.000 | 123.9 | 0 |
| optional | 1.000 | 1.000 | 0.000 | 33.2 | 0 |
| alternation | 1.000 | 1.000 | 0.000 | 26.2 | 0 |
| prefix_repeat | 1.000 | 0.000 | 1.000 | 770.0 | 0 |
| repeat_optional | 1.000 | 0.000 | 1.000 | 8.2 | 0 |
| prefix_alt | 1.000 | 0.438 | 0.000 | 9.7 | 0 |
| optional_internal | 1.000 | 1.000 | 0.000 | 126.8 | 0 |
| repeat_extra | 1.000 | 1.000 | 1.000 | 131.6 | 0 |

With gold facts union the top 8, atomic oracle is 1.000 and held-out oracle is 1.000. Across all four conditions the compose oracle is Gold facts 1.000, Gold + 1 distractor/head 1.000, Gold + 2 distractors/head 1.000, Gold ∪ model top 8 1.000. Truncated searches: 0.

The symbolic chart assembles a probe-matching program whenever the gold local facts are present, distractors included. Held-out local recall is still the open number. Scope relations between spans are the next place to train, and the recognizer weights stay as they are until that training starts. A pretrained backbone and BENCH-C stay off.

