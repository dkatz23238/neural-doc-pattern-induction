# Residual override

The old code-length rule picks R0 from the original 8-candidate neighborhood. A tiny model then scores the other candidates in the frozen structural 16 by Δ = hidden accuracy(Ri) − hidden accuracy(R0). It replaces R0 only when the best predicted Δ clears a threshold. False overrides are weighted more heavily than missed gains.

Training uses document compositions. Validation uses the same families with held-out prefixes, separators, and widths. Pooled and cross-attention models are compared on that validation split. BENCH-C is scored once. BENCH-A is not used.

## Reading

The val-selected residual model is cross (λ=5.0, epoch 2, threshold 0.2). On BENCH-C it scores 96.9% with regressions 11.9%. The old rule scores 96.2% (regressions 0.7%). Oracle on the same alternatives is 99.5%.

This stops the neural selector. Accuracy is at most 97%, or regressions exceed 3%.

On the held-out formats the same checkpoint scores 98.0% with regressions 1.8%. That margin did not survive the shift to BENCH-C.

The pooled model, which lost the validation comparison, scores 96.5% on BENCH-C with regressions 8.1%.

Too-broad failures move from 67.1% with the old rule to 97.5% (oracle 99.6%, n=10). Character-class columns move from 98.2% to 90.3% (n=27).

Epoch and threshold were frozen on held-out formatting variants before BENCH-C was scored. Training compositions and BENCH-C still share document families, so this number is an upper bound on transfer.

## Protocol

Selected `cross` at epoch 2 with threshold 0.2 and λ=5.0. Validation accuracy 98.0%, regressions 1.8%. Parameters: 619010.

## OOD-val

Held-out formatting variants. Used to choose epoch, threshold, and architecture.

| System | Semantic accuracy | Macro | Improved | Regressed | Overrides | False overrides |
| --- | --- | --- | --- | --- | --- | --- |
| heuristic | 86.1% | 86.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| old code-length rule | 90.0% | 89.9% | 21.2% | 0.2% | 0.0% | 0.0% |
| cross residual | 98.0% | 98.0% | 35.0% | 1.8% | 42.8% | 7.2% |
| pooled residual | 96.9% | 96.9% | 31.5% | 1.8% | 31.0% | 7.2% |
| oracle on R0 and the alternatives | 99.1% | 99.1% | 40.0% | 0.0% | 30.8% | 0.0% |

## C

Untouched document tasks, scored once. Families overlap the training compositions.

| System | Semantic accuracy | Macro | Improved | Regressed | Overrides | False overrides |
| --- | --- | --- | --- | --- | --- | --- |
| heuristic | 93.2% | 92.1% | 0.0% | 0.0% | 0.0% | 0.0% |
| old code-length rule | 96.2% | 95.7% | 14.4% | 0.7% | 0.0% | 0.0% |
| cross residual | 96.9% | 96.3% | 24.9% | 11.9% | 44.8% | 14.3% |
| pooled residual | 96.5% | 95.8% | 18.4% | 8.1% | 18.8% | 8.7% |
| oracle on R0 and the alternatives | 99.5% | 99.5% | 39.2% | 0.0% | 28.0% | 0.0% |

Heuristic-failure families:

| Failure | n | heuristic | old rule | residual | oracle |
| --- | --- | --- | --- | --- | --- |
| success | 402 | 99.1% | 99.1% | 98.3% | 99.7% |
| optional region | 50 | 85.3% | 88.7% | 94.6% | 97.8% |
| variable-length inference | 34 | 81.4% | 82.0% | 88.5% | 99.7% |
| character-class inference | 27 | 80.6% | 98.2% | 90.3% | 99.5% |
| too narrow | 22 | 60.6% | 98.8% | 97.4% | 100.0% |
| too_broad | 10 | 60.8% | 67.1% | 97.5% | 99.6% |
| mixed population | 4 | 63.5% | 91.1% | 100.0% | 100.0% |
| too broad | 3 | 68.1% | 70.8% | 92.4% | 100.0% |
| OCR noise | 1 | 79.2% | 100.0% | 100.0% | 100.0% |
| alternation | 1 | 89.6% | 89.6% | 91.7% | 100.0% |

## D

Sample-size curve. Not used to fit or to choose the threshold.

| System | Semantic accuracy | Macro | Improved | Regressed | Overrides | False overrides |
| --- | --- | --- | --- | --- | --- | --- |
| heuristic | 89.8% | 89.8% | 0.0% | 0.0% | 0.0% | 0.0% |
| old code-length rule | 93.8% | 93.8% | 16.6% | 0.0% | 0.0% | 0.0% |
| cross residual | 90.6% | 90.6% | 21.9% | 13.8% | 34.1% | 14.4% |
| pooled residual | 90.9% | 90.9% | 21.6% | 11.6% | 20.6% | 12.2% |
| oracle on R0 and the alternatives | 97.8% | 97.8% | 40.3% | 0.0% | 26.2% | 0.0% |
