## Visible coverage

The model and the grammar stay frozen. Alignment facts join a nonempty chart when they name a width, a literal, an optional region, or a binary alternation the frozen hypotheses do not already state. Equivalent concatenations collapse while the chart is built: adjacent copies, adjacent classes, a one-character class repeat, and an alternation branch that adds no strings. A chart the alignment does not extend is unchanged.

Exact tasks 320. Empty charts 0. Exact-slice synthesis oracle 0.922 (295/320). Previously perfect programs kept 113. Formerly empty charts that contain a hidden-set-perfect program 110. Of the 89 exact tasks that already had a program and no hidden-set-perfect one, 72 now contain one.

The other 25 exact misses, 17 from that 89 and 8 from the formerly empty charts, are positive-only identifiability failures. A bound endpoint, an alternative literal, or an optional text the gold uses does not appear in the four supports. None of the 25 is a visible structure the chart still fails to build. The recoverable-from-visible-support oracle is 0.922, the same number as the synthesis oracle.

| Missing fact | Tasks | What the four supports omit |
| --- | ---: | --- |
| unobserved bound | 16 | one endpoint of a class width, such as `{6,17}`, `{7,8}`, or `{1,3}` |
| unobserved branch | 8 | a literal alternative, such as a currency code, `N/A`, or a size |
| unobserved optional | 1 | an optional prefix that is present on every support |

Exact selector accuracy 0.966. Selector accuracy given a hidden-set-perfect program 0.974 on all 295 such tasks. Selector timeouts 0. The largest chart holds 36720 programs. Charts of that size are grouped by the regex they render, so distinct widths stay apart and the 90-second alarm does not fire. Headline selector accuracy, with these exact tasks rescored and every non-exact task left as scored, is 0.857.
