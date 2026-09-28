# Needle CPU

The scope recognizer, the version-space chart, and the witness head stay frozen. A contingent delimiter, a punctuation mark that some supports omit, adds an optional suffix when it is the last separator. Shared separators do not fire. The witness head is not retrained.

Parameters: 1226128 (102741 in the witness head).

| Family | Semantic top-1 | Semantic accuracy | Version-space oracle | Languages |
| --- | --- | --- | --- | --- |
| primitive | 1.000 | 1.000 | 1.000 | 137.4 |
| repeat | 1.000 | 1.000 | 1.000 | 421.1 |
| optional | 1.000 | 1.000 | 1.000 | 70.4 |
| alternation | 1.000 | 1.000 | 1.000 | 75.0 |
| prefix_repeat | 1.000 | 1.000 | 1.000 | 718.3 |
| repeat_optional | 1.000 | 1.000 | 1.000 | 4.6 |
| prefix_alt | 1.000 | 1.000 | 1.000 | 91.9 |
| optional_internal | 1.000 | 1.000 | 1.000 | 613.5 |
| repeat_extra | 1.000 | 1.000 | 1.000 | 547.7 |

Atomic semantic top-1 1.000. Held-out semantic top-1 1.000. Held-out semantic accuracy 1.000. Held-out version-space oracle 1.000. Worst held-out family prefix_repeat 1.000. Renaming invariance 1.000.

semantic top-1 cleared the stretch line on held-out composition.

## Stress

Eight draws per generator setting. Stability stays on. Core cells are two supports, probes at repeat depth 6–7, a within-class rename onto a held-out mark, and width-2 supports whose language still allows a width-1 piece. A lower-case prefix and a prefix-plus-repeat-plus-optional tail are reported beside those cells.

| Cell | Role | Semantic top-1 | Oracle | Languages | Max languages |
| --- | --- | --- | --- | --- | --- |
| few_support | core | 1.000 | 1.000 | 1037.2 | 8708 |
| few_support / prefix_repeat | core | 1.000 | 1.000 | 3299.6 | 8708 |
| few_support / repeat_optional | core | 1.000 | 1.000 | 3.4 | 5 |
| few_support / prefix_alt | core | 1.000 | 1.000 | 76.1 | 96 |
| few_support / optional_internal | core | 1.000 | 1.000 | 221.2 | 222 |
| few_support / repeat_extra | core | 1.000 | 1.000 | 1585.8 | 2799 |
| long_probe | core | 1.000 | 1.000 | 474.6 | 983 |
| rename | core | 1.000 | 1.000 | 390.2 | 1190 |
| rename / prefix_repeat | core | 1.000 | 1.000 | 680.1 | 1190 |
| rename / repeat_optional | core | 1.000 | 1.000 | 5.4 | 16 |
| rename / prefix_alt | core | 1.000 | 1.000 | 85.8 | 109 |
| rename / optional_internal | core | 1.000 | 1.000 | 743.1 | 1151 |
| rename / repeat_extra | core | 1.000 | 1.000 | 436.5 | 745 |
| adversarial_width | core | 1.000 | 1.000 | 1213.2 | 2388 |
| lower_prefix | report | 1.000 | 1.000 | 463.7 | 2495 |
| lower_prefix / prefix_repeat | report | 1.000 | 1.000 | 845.5 | 2495 |
| lower_prefix / prefix_alt | report | 1.000 | 1.000 | 81.9 | 134 |
| nested | report | 0.500 | 0.500 | 2.0 | 6 |

Core semantic top-1 1.000. Core oracle 1.000. Worst core cell few_support 1.000.

