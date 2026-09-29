# Renaming equivariance

The question is whether membership depends on symbol identity or on relations inside the task.
A within-class rename may send `-` to `§` and `A` to `Q`. The label stays the same.
Invariance is `1 - mean |p(S, x) - p(π(S), π(x))|`.
The relational encoder sees structural class, a task-local symbol id, and alignment relations. It does not see raw codepoints.
BENCH-C is not scored.

Byte-identity model: 1877186 parameters, diagnostic only.
Relational model: 973121 parameters. Selected epoch: 6.

## Byte-identity model, familiar alphabet only

| Slice | Balanced | AUROC | Member acceptance | Invariance | Renamed balanced |
| --- | --- | --- | --- | --- | --- |
| simple | 0.794 | 0.853 | 0.900 | 0.749 | 0.544 |
| clean | 0.725 | 0.787 | 1.000 | 0.818 | 0.512 |
| structural | 0.700 | 0.724 | 0.950 | 0.794 | 0.500 |

On the renamed clean probe the byte model still accepts true members (1.000) and almost stops rejecting near misses (negative rejection 0.025, was 0.450). Balanced accuracy falls from 0.725 to 0.512 and AUROC from 0.787 to 0.524.

## Relational model

| Slice | Balanced | AUROC | Member acceptance | Invariance | Renamed balanced | Renamed member acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| simple | 0.925 | 0.980 | 0.925 | 1.000 | 0.925 | 0.925 |
| clean | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| structural | 0.762 | 0.846 | 0.700 | 1.000 | 0.762 | 0.700 |

## Gates

These are the early gates for a relational backbone. They are not BENCH-C.

| Gate | Target | Result | Met |
| --- | --- | --- | --- |
| simple training alphabet | >0.90 balanced | 0.925 | yes |
| symbol-renamed simple score stays within 0.03 | same | 0.925 | yes |
| unseen-delimiter clean probe | >0.85 balanced | 1.000 | yes |
| held-out-byte true-member acceptance | >0.85 | 1.000 | yes |
| renamed structural AUROC | >0.90 | 0.846 | no |

## Reading

Renaming is no longer able to move the relational model. Invariance is 1.000 on every slice: `p(S, x)` and `p(π(S), π(x))` are the same number, including when `-` becomes `§`.

The byte-identity diagnostic does move. On letter-separator-digit tasks it scores 0.725 balanced (AUROC 0.787) with the training separator and 0.512 (AUROC 0.524) after that separator leaves the alphabet. Invariance there is 0.818. Negative rejection falls from 0.450 to 0.025. Familiar simple languages fall from 0.794 to 0.544 under a within-class rename.

The relational model scores 0.925 balanced (AUROC 0.980) on simple languages and the same 0.925 after the rename. The clean delimiter probe, before and after the held-out separator, is 1.000 balanced with true-member acceptance 1.000. That is the symmetry the byte model did not have.

Repeats, optional tails, and alternations are unchanged by the rename as well, and they are not solved. Renamed structural accuracy is 0.762 balanced and AUROC is 0.846, short of the 0.90 mark. A regex decoder stays off until that composition gate moves. The remaining gap is structure, not which symbol plays the separator.

Epochs

- epoch 1 byte prefix-class: loss 1.077, clean renamed 0.475
- epoch 2 byte delimited: loss 0.926, clean renamed 0.512
- epoch 3 relational mixed-1: loss 1.106, simple 0.800, clean renamed 0.875
- epoch 4 relational mixed-2: loss 0.875, simple 0.906, clean renamed 1.000
- epoch 5 relational mixed-3: loss 0.795, simple 0.894, clean renamed 1.000
- epoch 6 relational mixed-4: loss 0.767, simple 0.925, clean renamed 1.000
