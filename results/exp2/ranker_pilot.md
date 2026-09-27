# CPU ranker pilot

The ranker sees visible strings and the same 8 symbolic candidates as the oracle
experiment. Labels are a softmax over hidden semantic accuracy with temperature
0.05. BENCH-C is not used for training or epoch selection.
The pooled model replaces candidate-to-example attention with an MLP on the mean
of the example vectors, sized to about the same parameter count.

## Reading

On held-out BENCH-A shapes, the code-length rule scores 86.5% and the 8-candidate oracle scores 90.3%. Cross-attention, trained on BENCH-A dev, scores 87.0%. The pooled model scores 87.4%.

On untouched BENCH-C, the rule scores 96.2% and the oracle scores 97.4%. The structural cross-attention model scores 93.9% (regressions 5.8%). The document-trained cross-attention model scores 94.3% (regressions 16.1%).

Neither training distribution beats the code-length rule by a point on its decision split. Train loss can still fall. The ranking signal in this candidate set is not something these tiny nets pick up beyond compression, so a larger GPU run of the same ranker is not justified.

Cross-attention and the pooled model are within half a point on BENCH-A test. Candidate-specific attention is not earning its keep at this size.

## Systems

### bench_a_test

Held-out AST shapes. This is the structural decision split.

| System | Semantic accuracy | Macro | Improved | Regressed | Unchanged |
| --- | --- | --- | --- | --- | --- |
| heuristic | 84.0% | 84.0% | 0.0% | 0.0% | 100.0% |
| rules | 86.5% | 86.5% | 23.0% | 6.5% | 70.5% |
| oracle_k | 90.3% | 90.3% | 41.5% | 0.0% | 58.5% |
| cross_structural | 87.0% | 87.0% | 28.0% | 20.0% | 52.0% |
| pooled_structural | 87.4% | 87.4% | 30.5% | 19.5% | 50.0% |
| cross_document | 84.3% | 84.3% | 22.5% | 29.5% | 48.0% |

### bench_a_val

Used to choose the epoch for the structural models.

| System | Semantic accuracy | Macro | Improved | Regressed | Unchanged |
| --- | --- | --- | --- | --- | --- |
| heuristic | 84.0% | 84.0% | 0.0% | 0.0% | 100.0% |
| rules | 88.4% | 88.4% | 31.0% | 5.5% | 63.5% |
| oracle_k | 91.1% | 91.1% | 46.0% | 0.0% | 54.0% |
| cross_structural | 88.8% | 88.8% | 34.5% | 11.5% | 54.0% |
| pooled_structural | 88.7% | 88.7% | 35.5% | 11.0% | 53.5% |
| cross_document | 86.0% | 86.0% | 30.0% | 32.0% | 38.0% |

### bench_a_dev

Training data for the structural models.

| System | Semantic accuracy | Macro | Improved | Regressed | Unchanged |
| --- | --- | --- | --- | --- | --- |
| heuristic | 84.6% | 84.6% | 0.0% | 0.0% | 100.0% |
| rules | 87.7% | 87.7% | 22.6% | 5.2% | 72.1% |
| oracle_k | 90.8% | 90.8% | 43.1% | 0.0% | 56.9% |
| cross_structural | 88.8% | 88.8% | 29.8% | 12.2% | 58.0% |
| pooled_structural | 88.5% | 88.5% | 30.4% | 14.4% | 55.2% |
| cross_document | 85.2% | 85.2% | 23.9% | 31.4% | 44.8% |

### bench_c

Untouched document tasks. Not used to train or to stop.

| System | Semantic accuracy | Macro | Improved | Regressed | Unchanged |
| --- | --- | --- | --- | --- | --- |
| heuristic | 93.2% | 92.1% | 0.0% | 0.0% | 100.0% |
| rules | 96.2% | 95.7% | 14.4% | 0.7% | 84.8% |
| oracle_k | 97.4% | 97.1% | 24.9% | 0.0% | 75.1% |
| cross_structural | 93.9% | 92.9% | 8.3% | 5.8% | 85.9% |
| pooled_structural | 94.6% | 93.8% | 10.5% | 2.9% | 86.6% |
| cross_document | 94.3% | 93.5% | 15.2% | 16.1% | 68.8% |

### bench_d

Same schemas at several sample sizes. Not used to train or to stop.

| System | Semantic accuracy | Macro | Improved | Regressed | Unchanged |
| --- | --- | --- | --- | --- | --- |
| heuristic | 89.8% | 89.8% | 0.0% | 0.0% | 100.0% |
| rules | 93.8% | 93.8% | 16.6% | 0.0% | 83.4% |
| oracle_k | 96.8% | 96.8% | 29.4% | 0.0% | 70.6% |
| cross_structural | 92.3% | 92.3% | 12.8% | 5.3% | 81.9% |
| pooled_structural | 93.0% | 93.0% | 18.1% | 6.9% | 75.0% |
| cross_document | 92.9% | 92.9% | 16.6% | 9.7% | 73.8% |

### bench_b_val

Used to choose the epoch for the document model. Families overlap BENCH-C.

| System | Semantic accuracy | Macro | Improved | Regressed | Unchanged |
| --- | --- | --- | --- | --- | --- |
| heuristic | 91.9% | 90.6% | 0.0% | 0.0% | 100.0% |
| rules | 95.9% | 95.6% | 19.9% | 0.3% | 79.8% |
| oracle_k | 96.8% | 96.5% | 26.8% | 0.0% | 73.2% |
| cross_structural | 93.7% | 92.8% | 15.6% | 5.0% | 79.5% |
| pooled_structural | 93.9% | 93.1% | 16.6% | 4.0% | 79.5% |
| cross_document | 96.6% | 96.3% | 25.8% | 1.3% | 72.9% |

## BENCH-C by heuristic failure

| Failure | n | heuristic | rules | oracle_k | cross_structural | pooled_structural | cross_document |
| --- | --- | --- | --- | --- | --- | --- | --- |
| success | 402 | 99.1% | 99.1% | 99.3% | 99.0% | 99.0% | 98.1% |
| optional region | 50 | 85.3% | 88.7% | 91.2% | 85.5% | 86.8% | 87.2% |
| variable-length inference | 34 | 81.4% | 82.0% | 92.3% | 84.3% | 84.3% | 81.1% |
| character-class inference | 27 | 80.6% | 98.2% | 99.0% | 84.7% | 89.1% | 91.0% |
| too narrow | 22 | 60.6% | 98.8% | 99.6% | 67.8% | 75.4% | 85.3% |
| too_broad | 10 | 60.8% | 67.1% | 67.1% | 66.5% | 66.5% | 65.8% |
| mixed population | 4 | 63.5% | 91.1% | 91.1% | 66.7% | 66.7% | 73.4% |
| too broad | 3 | 68.1% | 70.8% | 70.8% | 70.8% | 70.8% | 70.8% |
| OCR noise | 1 | 79.2% | 100.0% | 100.0% | 79.2% | 79.2% | 100.0% |
| alternation | 1 | 89.6% | 89.6% | 89.6% | 89.6% | 89.6% | 89.6% |

## Training

`cross_structural` has 326497 parameters, selected epoch 8 of 16 by validation semantic accuracy (88.8%), in 95s.

| Epoch | Train loss | Train semantic | Val semantic | Train top-1 |
| --- | --- | --- | --- | --- |
| 1 | 1.835 | 85.9% | 87.0% | 33.1% |
| 2 | 1.804 | 87.2% | 88.1% | 42.9% |
| 3 | 1.787 | 87.6% | 88.4% | 42.6% |
| 4 | 1.774 | 87.8% | 88.2% | 41.8% |
| 5 | 1.761 | 88.1% | 88.5% | 45.0% |
| 6 | 1.753 | 88.3% | 88.6% | 39.9% |
| 7 | 1.739 | 88.6% | 88.5% | 45.6% |
| 8 | 1.730 | 88.6% | 88.8% | 41.2% |
| 9 | 1.719 | 88.7% | 88.6% | 41.5% |
| 10 | 1.705 | 89.2% | 88.5% | 44.1% |
| 11 | 1.693 | 89.4% | 88.4% | 44.0% |
| 12 | 1.679 | 89.5% | 88.3% | 46.4% |
| 13 | 1.664 | 89.7% | 87.9% | 45.8% |
| 14 | 1.657 | 89.8% | 88.4% | 47.6% |
| 15 | 1.646 | 90.0% | 88.2% | 48.5% |
| 16 | 1.633 | 90.1% | 87.9% | 50.4% |

`pooled_structural` has 326401 parameters, selected epoch 9 of 17 by validation semantic accuracy (88.7%), in 114s.

| Epoch | Train loss | Train semantic | Val semantic | Train top-1 |
| --- | --- | --- | --- | --- |
| 1 | 1.840 | 85.6% | 86.6% | 31.1% |
| 2 | 1.810 | 87.2% | 87.8% | 42.6% |
| 3 | 1.790 | 87.5% | 87.8% | 40.2% |
| 4 | 1.780 | 87.8% | 88.1% | 40.8% |
| 5 | 1.770 | 88.0% | 88.3% | 44.5% |
| 6 | 1.767 | 88.0% | 88.5% | 41.0% |
| 7 | 1.760 | 88.2% | 88.5% | 43.5% |
| 8 | 1.757 | 88.2% | 88.5% | 42.5% |
| 9 | 1.748 | 88.5% | 88.7% | 41.6% |
| 10 | 1.737 | 88.6% | 88.5% | 43.0% |
| 11 | 1.728 | 88.7% | 88.4% | 42.2% |
| 12 | 1.723 | 88.7% | 88.6% | 42.2% |
| 13 | 1.714 | 89.1% | 88.3% | 44.4% |
| 14 | 1.702 | 89.1% | 88.5% | 43.4% |
| 15 | 1.697 | 89.2% | 88.4% | 45.5% |
| 16 | 1.686 | 89.3% | 88.1% | 46.9% |
| 17 | 1.677 | 89.6% | 88.4% | 47.6% |

`cross_document` has 326497 parameters, selected epoch 9 of 17 by validation semantic accuracy (96.6%), in 174s.

| Epoch | Train loss | Train semantic | Val semantic | Train top-1 |
| --- | --- | --- | --- | --- |
| 1 | 1.923 | 95.3% | 95.1% | 19.2% |
| 2 | 1.798 | 96.5% | 96.2% | 37.1% |
| 3 | 1.769 | 96.7% | 96.4% | 44.7% |
| 4 | 1.751 | 97.0% | 96.6% | 45.9% |
| 5 | 1.741 | 97.1% | 96.6% | 50.2% |
| 6 | 1.734 | 97.1% | 96.5% | 46.9% |
| 7 | 1.732 | 97.1% | 96.5% | 46.9% |
| 8 | 1.725 | 97.2% | 96.6% | 48.6% |
| 9 | 1.723 | 97.1% | 96.6% | 45.8% |
| 10 | 1.720 | 97.2% | 96.6% | 52.7% |
| 11 | 1.717 | 97.2% | 96.6% | 47.7% |
| 12 | 1.717 | 97.2% | 96.6% | 47.6% |
| 13 | 1.715 | 97.2% | 96.6% | 46.6% |
| 14 | 1.715 | 97.2% | 96.6% | 46.7% |
| 15 | 1.713 | 97.2% | 96.6% | 48.0% |
| 16 | 1.712 | 97.2% | 96.6% | 44.2% |
| 17 | 1.711 | 97.2% | 96.6% | 51.3% |

## Protocol

AdamW learning rate 0.0003, weight decay 0.01, gradient clip 1, batch size 32, soft-label temperature 0.05, maximum 40 epochs, patience 8 on validation semantic accuracy. Byte length cap 128. Truncated inputs: 31.

Structural models train on BENCH-A dev and stop on BENCH-A val semantic accuracy. The document model trains on the BENCH-B hash fold used in the oracle experiment and stops on the held-out fifth of BENCH-B. Soft targets use hidden semantic accuracy. Model inputs are only the visible strings and the candidate regex text.
