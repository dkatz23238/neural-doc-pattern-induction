## In-fragment transfer

The model and the grammar stay frozen. A nonempty chart is not rebuilt. An empty chart gains facts from an alignment of class runs: a shared text is a copy, a varying run is that class at the observed widths, a fixed run sequence that some supports omit is optional, and two unrelated skeletons are an alternation. Class width is not held to the repeat cap.

Exact tasks 320. Empty charts before 118, after 0 (0.000). Exact-slice synthesis oracle 0.697. Exact-slice selector accuracy 0.942. Headline selector accuracy, with previously empty tasks rescored and every other task left as scored, 0.843.

Of the 118 charts that were empty, 110 now contain a hidden-set-perfect program. The other 8 have a program and miss the hidden set because a width, a prefix, or a suffix never appeared in the four supports: five bank references, one phone width, one order-id prefix, and one section suffix.

Rescored 118 previously empty exact tasks in 2063s. Timeouts 12. Each of those 12 already had a hidden-set-perfect program; equivalence grouping did not finish, so the selector is counted as an abstention. On the 106 that finished, selector accuracy is 0.964.

| Alignment | Empty exact tasks | Filled | Hidden-set perfect | Selector accuracy |
| --- | --- | --- | --- | --- |
| equal | 101 | 101 | 94 | 0.918 |
| optional_prefix | 4 | 4 | 3 | 0.833 |
| optional_suffix | 9 | 9 | 9 | 0.914 |
| two_cluster | 4 | 4 | 4 | 0.979 |
