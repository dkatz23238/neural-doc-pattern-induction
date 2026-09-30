# Fitted confidence

`needle()`'s confidence is a logistic model over signals it already computes. The label is that the code-length winner scores at least 0.99 on a BENCH-C task's hidden strings; 353 of 554 tasks do. The recognizer's signals count only when it finished within the 5-second limit `needle()` uses, measured from the timings in `confidence_features.jsonl` (three workers sharing the CPU, so slower than a single call). Every score below is out of fold: all tasks of one domain (33 domains) are held out together, and the ridge strength is chosen inside the remaining domains. The weights in `confidence_fit.json` are fit on every task with ridge 100. BENCH-E is not used.

| Score | AUROC | Brier | Log loss | ECE |
| --- | ---: | ---: | ---: | ---: |
| always the base rate | 0.500 | 0.231 | 0.655 | 0.000 |
| hand-set v1 formula | 0.582 | 0.259 | 0.888 | 0.175 |
| fitted, held-out domains | 0.672 | 0.217 | 0.630 | 0.073 |

Reliability of the held-out fitted score:

| Score | Tasks | Mean score | Winner correct |
| --- | ---: | ---: | ---: |
| 0.00–0.30 | 20 | 0.220 | 0.650 |
| 0.30–0.50 | 95 | 0.429 | 0.421 |
| 0.50–0.70 | 204 | 0.611 | 0.564 |
| 0.70–0.85 | 202 | 0.757 | 0.777 |
| 0.85–0.95 | 33 | 0.885 | 0.848 |

Share correct among the most confident tasks:

| Kept | Fitted | Hand-set |
| --- | ---: | ---: |
| 10% | 0.818 | 0.818 |
| 25% | 0.761 | 0.674 |
| 50% | 0.794 | 0.708 |
| 75% | 0.689 | 0.672 |
| 100% | 0.637 | 0.637 |

Weights on standardized features:

| Weight | Feature |
| ---: | --- |
| +0.257 | gap in bits |
| -0.245 | regex #ranges |
| -0.229 | bits per character |
| +0.226 | network agrees |
| +0.188 | regex has open repeat |
| +0.142 | log max length |
| +0.121 | #rule alternatives |
| -0.096 | log #network candidates |
| -0.094 | runner-up within 1 bit |
| -0.090 | no stricter runner-up |
| -0.067 | log regex length |
| -0.065 | regex has \| |
| +0.062 | regex #classes |
| +0.041 | network answered |
| +0.037 | network's own margin |
| +0.025 | example over 32 chars |
| +0.000 | #examples |
