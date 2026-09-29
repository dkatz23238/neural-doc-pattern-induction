# Compositional language induction

The question is whether a symbol-invariant model can learn repetition, optionality, and alternation, then apply them in combinations it was not trained on.
Relational character features stay as they were: structural class, a task-local symbol id, and alignment relations. No codepoint channel is added.
Class runs and adjacent pairs become spans. A second transformer relates those spans. Generator groups supervise span identity, kind, boundaries, and operator role.
Training mixes primitive prefix+class, repeats of 1–3, an optional tail, and a bare alternation. Checkpoint selection uses only those four.
Repeat extrapolation asks for 4–5 repetitions. Optional extrapolation moves the optional group inside the string. Held-out compositions are prefix+repeat, repeat+optional, and prefix+alternation.
Invariance is `1 - mean |p(S, x) - p(π(S), π(x))|`. BENCH-C is not scored.

Parameters: 945464. Selected step: 1200.

## Atomic operators

| Slice | Balanced | AUROC | Member acceptance | Invariance | Renamed AUROC |
| --- | --- | --- | --- | --- | --- |
| primitive | 0.988 | 0.996 | 1.000 | 1.000 | 0.996 |
| repeat | 0.975 | 0.999 | 0.963 | 1.000 | 0.999 |
| optional | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| alternation | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Extrapolation

| Slice | Balanced | AUROC | Member acceptance | Invariance | Renamed AUROC |
| --- | --- | --- | --- | --- | --- |
| repeat_extra | 0.775 | 0.935 | 0.613 | 1.000 | 0.935 |
| optional_internal | 0.750 | 1.000 | 0.500 | 1.000 | 1.000 |
| prefix_alt | 0.844 | 0.987 | 0.700 | 1.000 | 0.987 |

## Held-out compositions

| Slice | Balanced | AUROC | Member acceptance | Invariance | Renamed AUROC |
| --- | --- | --- | --- | --- | --- |
| prefix_repeat | 0.469 | 0.548 | 0.188 | 1.000 | 0.548 |
| repeat_optional | 0.669 | 0.710 | 0.350 | 1.000 | 0.710 |
| prefix_alt | 0.844 | 0.987 | 0.700 | 1.000 | 0.987 |

## Gates

| Gate | Target | Result | Met |
| --- | --- | --- | --- |
| primitive AUROC | >0.93 | 0.996 | yes |
| repeats AUROC | >0.90 | 0.999 | yes |
| optional spans AUROC | >0.90 | 1.000 | yes |
| alternations AUROC | >0.90 | 1.000 | yes |
| held-out compositions AUROC | >0.90 | 0.748 | no |
| renaming invariance | ~1.00 | 1.000 | yes |

## Reading

A within-class rename leaves every probability in place. Invariance is 1.000 on every slice, including when a separator becomes a held-out mark.

The atomic operators clear the AUROC gates. Primitive prefix-plus-class is AUROC 0.996 (balanced 0.988). Repeats of 1–3 are AUROC 0.999 (balanced 0.975). Optional tails are AUROC 1.000 (balanced 1.000). Bare alternations are AUROC 1.000 (balanced 1.000).

Longer repeats still rank above broken strings of the same depth: AUROC 0.935, negative rejection 0.938. True-member acceptance falls from 0.963 inside the training depths to 0.613, and balanced accuracy from 0.975 to 0.775. An optional group in the middle, absent from training, keeps AUROC 1.000 and negative rejection 1.000. True-member acceptance there is 0.500, so balanced accuracy is 0.750.

The three held-out mixtures land in different places. Prefix plus alternation is AUROC 0.987 (balanced 0.844). Repeat plus an optional tail is AUROC 0.710 (balanced 0.669, true-member acceptance 0.350). Prefix plus repeat is AUROC 0.548 (balanced 0.469, true-member acceptance 0.188). The mean of those three AUROCs is 0.748. The composition gate stays open, so the regex decoder stays off.

Steps

- step 100: loss 1.000, select AUROC 0.932, primitive 0.958, repeat 0.915, optional 0.938, alternation 0.916
- step 200: loss 0.433, select AUROC 0.941, primitive 0.983, repeat 0.931, optional 0.949, alternation 0.901
- step 300: loss 0.138, select AUROC 0.964, primitive 0.986, repeat 0.942, optional 0.997, alternation 0.932
- step 400: loss 0.251, select AUROC 0.975, primitive 0.992, repeat 0.949, optional 0.999, alternation 0.959
- step 500: loss 0.619, select AUROC 0.970, primitive 0.993, repeat 0.942, optional 0.995, alternation 0.949
- step 600: loss 0.090, select AUROC 0.988, primitive 0.996, repeat 0.955, optional 1.000, alternation 1.000
- step 700: loss 0.352, select AUROC 0.983, primitive 0.994, repeat 0.936, optional 1.000, alternation 1.000
- step 800: loss 0.010, select AUROC 0.990, primitive 0.993, repeat 0.967, optional 1.000, alternation 1.000
- step 900: loss 0.001, select AUROC 0.991, primitive 0.999, repeat 0.964, optional 1.000, alternation 1.000
- step 1000: loss 0.001, select AUROC 0.989, primitive 0.998, repeat 0.957, optional 1.000, alternation 1.000
- step 1100: loss 0.001, select AUROC 0.990, primitive 0.993, repeat 0.965, optional 1.000, alternation 1.000
- step 1200: loss 0.001, select AUROC 0.992, primitive 0.992, repeat 0.978, optional 1.000, alternation 1.000
