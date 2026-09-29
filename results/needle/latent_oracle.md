## Latent candidate oracle

The support-identifiable program is what the four examples justify. BENCH-C often asks for the latent rule those examples did not reveal. This pass adds generic rewrites — wider bounds, conventional numeric intervals, natural character superclasses, open repeats, alternations lifted to a class, and field structure read off the supports, including thousands groups, repeated fields, and optional affixes — and asks whether the hidden language is then in the set. No prior is trained. BENCH-E stays sealed.

Current synthesis oracle on 554 tasks is 0.796 (441/554). Expanded synthesis oracle is 0.915 (507/554). The rewrites recover 66 tasks the current chart missed. On the tasks that were reassembled, the mean chart has 1496 programs and the mean latent rewrite set has 286. Assembly timeouts: 0.

The support-identifiable program is hidden-perfect on 68 tasks. The other 486 fall into the ambiguity classes below. Those classes are the curriculum for a later prior, not extra benchmark answers.

| Ambiguity | Tasks |
| --- | ---: |
| unseen width/bound | 112 |
| unseen class member | 270 |
| unseen alternative | 42 |
| unseen optional branch | 1 |
| semantic numeric interval | 39 |
| open quantifier | 10 |
| other | 12 |

| Metric | Value |
| --- | --- |
| Current synthesis oracle | 0.796 |
| Expanded synthesis oracle | 0.915 |
| Recovered by expansion | 66 |
| Mean chart on reassembled tasks | 1496 |
| Mean latent rewrites on reassembled tasks | 286 |

The expanded oracle is not yet above 0.98, so these rewrites are not a basis for training a language prior. What remains is mostly closed lexicons, supports clipped at 32 characters, and optional branches that the four examples never show. BENCH-E stays sealed.

