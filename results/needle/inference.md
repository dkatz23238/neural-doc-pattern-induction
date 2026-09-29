# Inference profile

The decision is unchanged. On all 554 BENCH-C tasks, `symbolic_pick` returns the same pattern it returned before this pass, and the semantic accuracy is again **0.9624699157641396**. BENCH-E was not reopened.

The time is in the recognizer chart, not in the symbolic selector. A cold profile of `needle()` on this CPU, before the caches:

| Stage | invoice | version | section | filename | purchase order | customer |
| --- | --- | --- | --- | --- | --- | --- |
| recognizer | 0.63s | 6.92s | 0.73s | 1.64s | 0.14s | 0.18s |

Candidate generation and the code-length score are 2–4 ms. Regex compilation of that neighborhood is inside those milliseconds. The chart was the cost: version strings assemble 17,348 programs, then equivalence, stability, and grammar scoring walk that list. Membership keys were fresh `describe()` strings, grammar scoring flattened every tree twice, and packing was repeated for shared subtrees.

The hot path now keeps a canonical AST id on each program, memos subtree matches, packs, regex fragments, and grammar scores, and scores the grammar only inside the witness band. Support class expansions are cached. Equal trees share one id, so a subtree that already matched is not walked again.

After that, the same recognizer hypothesis is returned and the full call is:

| Call | Decision | `needle()` | Confidence | Ambiguous |
| --- | --- | --- | --- | --- |
| invoice | 3.4 ms | 0.25 s | 0.74 | yes |
| version | 2.3 ms | 1.69 s | 0.58 | yes |
| section | 1.8 ms | 0.33 s | 0.62 | yes |
| filename | 2.6 ms | 0.49 s | 0.75 | yes |
| purchase order | 1.6 ms | 0.13 s | 0.94 | no |
| customer | 2.4 ms | 0.16 s | 0.71 | yes |

Resident memory after the load and these calls is 355 MB, almost all of it the frozen recognizer. `scripts/profile_needle.py` reprints this table and checks that the six regexes are still the frozen ones.

Confidence multiplies three factors and does not choose the regex. The code-length margin saturates as the gap grows. The recognizer contributes its grammar gap only when its regex is the same language as the winner; a different language discounts the report and keeps it at or below 0.8. A runner-up inside one bit scales the report by 0.6 and caps it at 0.7. A spelling such as `PO\-` versus `PO-` counts as agreement. Alternatives are dropped when they are that same language, including an exact repeat written out as copies, and four semantically different runners remain.
