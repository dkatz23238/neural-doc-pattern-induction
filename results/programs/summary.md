# Execution-guided program synthesis

P1 keeps the symbol-invariant encoder and asks it to propose a program. Sequence, repeat, optional, and alternation are executed exactly. Literals are copied from a support span, so an unseen delimiter is retrieved rather than generated from a character vocabulary.

Training mixes the atomic languages with the D1 schemas. The eval compositions, including repeats of depth 4–5, are held out. A beam of 16 programs is executed on the supports; anything that misses a support is dropped. The reported top-1 is the remaining program with the best model score minus a small length prior.

The gates are renaming invariance about 1, support acceptance 1 after that filter, held-out semantic top-1 above 0.90, oracle@4 above 0.95, oracle@16 above 0.98, and repeat-depth top-1 above 0.90. Semantic agreement is execution on hidden positives and negatives, not exact AST text.

Parameters: 1230214. Selected step: 1000.

| Family | Semantic top-1 | Raw top-1 | Oracle@4 | Oracle@16 | Support acceptance |
| --- | --- | --- | --- | --- | --- |
| primitive | 0.792 | 0.792 | 0.792 | 0.792 | 0.792 |
| repeat | 0.792 | 0.792 | 0.792 | 0.792 | 0.792 |
| optional | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 |
| alternation | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| prefix_repeat | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| repeat_optional | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| prefix_alt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| optional_internal | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| repeat_extra | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 |

Atomic top-1 0.677. Held-out top-1 0.133. Held-out oracle@4 0.133. Held-out oracle@16 0.133. Repeat 4–5 top-1 0.667. Support acceptance 0.000. Renaming invariance 1.000. Mean programs in the beam 2.5.

No support-consistent program was found for prefix_repeat, repeat_optional, prefix_alt, optional_internal. Oracle@16 on those families is not a ranking among 16 candidates.

Where a program is returned, oracle@16 equals semantic top-1. Extra beam entries did not add a correct language.

The proposer is not putting a semantically correct program in the top 16 on the held-out languages. Alternation is solved and the atomic repeat is often exact, including the copied separator. The held-out mixtures are not. A pretrained backbone and BENCH-C stay off.

