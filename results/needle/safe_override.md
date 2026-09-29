# Safe correction of the symbolic selector

The default is the code-length rule. The alternative is the program Needle already returns for the same supports. The model sees the symbolic winner, that alternative, a summary of the supports, the frozen local evidence score of Needle's program, and the structural difference between the two patterns. It may replace the symbolic winner only when P(alternative is better) is above a threshold.

Training rows are disagreements only. A wrong override is weighted ten times a missed correction. The threshold is swept on held-out prefixes, separators, and widths. It has to keep validation accuracy at or above the symbolic score, hold the regression rate to 2%, and keep the easy family within 0.5% of the symbolic score. BENCH-C is scored once.

## Reading

The factory prior is negative for v1. Typed scales did not reliably beat the frozen evidence layer on held-out concepts (0.451 / 0.458 against evidence 0.472 / 0.458). This page asks a smaller question: keep the symbolic winner, and override it only when a model is sure Needle's existing program is better.

The val-selected threshold is 0.5 at epoch 6 (λ=10.0). On the held-out formats the override scores 94.5% against a symbolic 91.0%, with net 15 (fixed 15 minus regressions 0).

On BENCH-C the override scores 0.9633. The symbolic selector scores 0.9625. Needle's alternative scores 0.8896. The oracle union of those two scores 0.9731. Errors fixed 27, regressions 26 (4.7% of tasks), net 1.

The easy family (heuristic success, n=402) moves from 0.9909 to 0.9862 (drop 0.0047). The precommitted margin is 0.005.

This does not clear the v1 bar. The score rises by less than a tenth of a point, while the regression rate is 4.7%, past the 2% budget. It stays short of 96.7%. Ship the 96.2% symbolic selector together with the learned evidence and confidence machinery. Do not spend another search on the half point.

Needle returned no program within 25s on 12 BENCH-C tasks, so those rows kept the symbolic winner. This run's oracle union is the fresh Needle program, not a replay of the stored 97.7% selector pass.

Epoch and threshold were frozen on held-out formatting variants before BENCH-C was scored. Training compositions and BENCH-C share document families, so a gain here is an upper bound on transfer. BENCH-E was not scored. needle() is unchanged.

## Protocol

λ=10.0 fixed. Parameters 2145. Training tasks 320, disagreements 319, alternatives that improve the symbolic score 76. Validation tasks 80, disagreements 79. Selected epoch 6, threshold 0.5. Target band 96.7%. Evidence parameters 1158.

## Validation

Held-out formatting variants. Used to choose the epoch and the threshold.

| System | Semantic accuracy | Fixed | Regressed | Net | Overrides |
| --- | --- | --- | --- | --- | --- |
| symbolic selector | 91.0% | 0 | 0 | 0 | 0 |
| Needle alternative | 87.1% | 17 | 41 | -24 | 79 |
| learned override | 94.5% | 15 | 0 | 15 | 15 |
| oracle union | 94.8% | 17 | 0 | 17 | 17 |

## BENCH-C

Scored once, after the threshold was frozen. Regressions are drops from the symbolic winner.

| System | Semantic accuracy | Fixed | Regressed | Net | Overrides |
| --- | --- | --- | --- | --- | --- |
| symbolic selector | 96.2% | 0 | 0 | 0 | 0 |
| Needle alternative | 89.0% | 58 | 359 | -301 | 518 |
| learned override | 96.3% | 27 | 26 | 1 | 63 |
| oracle union | 97.3% | 58 | 0 | 58 | 65 |

Minimum bar (above the symbolic score, regression rate ≤ 2%, easy family within 0.5%): no. Target band (≥ 96.7% with the same constraints): no.

Heuristic-failure families:

| Failure | n | symbolic | override | oracle union |
| --- | --- | --- | --- | --- |
| success | 402 | 99.1% | 98.6% | 99.2% |
| optional region | 50 | 88.7% | 90.7% | 92.1% |
| variable-length inference | 34 | 82.0% | 82.8% | 87.6% |
| character-class inference | 27 | 98.2% | 97.5% | 99.3% |
| too narrow | 22 | 98.8% | 98.8% | 98.8% |
| too_broad | 10 | 67.1% | 79.4% | 81.7% |
| mixed population | 4 | 91.1% | 89.1% | 91.1% |
| too broad | 3 | 70.8% | 77.1% | 77.1% |
| OCR noise | 1 | 100.0% | 100.0% | 100.0% |
| alternation | 1 | 89.6% | 89.6% | 89.6% |

## Threshold sweep

Validation only.

| Threshold | Accuracy | Symbolic | Fixed | Regressed | Net | Eligible |
| --- | --- | --- | --- | --- | --- | --- |
| 0.2 | 87.2% | 91.0% | 17 | 40 | -23 | no |
| 0.3 | 92.0% | 91.0% | 17 | 27 | -10 | no |
| 0.4 | 93.9% | 91.0% | 17 | 7 | 10 | no |
| 0.5 | 94.5% | 91.0% | 15 | 0 | 15 | yes |
| 0.6 | 93.2% | 91.0% | 9 | 0 | 9 | yes |
| 0.7 | 92.4% | 91.0% | 5 | 0 | 5 | yes |
| 0.8 | 91.0% | 91.0% | 0 | 0 | 0 | yes |
| 0.9 | 91.0% | 91.0% | 0 | 0 | 0 | yes |
| 0.95 | 91.0% | 91.0% | 0 | 0 | 0 | yes |
| 0.99 | 91.0% | 91.0% | 0 | 0 | 0 | yes |
| 1.0 | 91.0% | 91.0% | 0 | 0 | 0 | yes |
