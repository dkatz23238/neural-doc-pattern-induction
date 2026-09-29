# Hierarchical group composition

The data-recipe study is the control. H0 is that span model. H1 and H2 keep its character features, span encoder, optimizer, 1200 steps, batch, checkpoint rule, and evaluation slices.
H1 pools spans into groups and mixes the groups with a transformer. H2 mixes them by typed edges: parent and child, next sibling, same repeated unit, and alignment from the query group to a support group with the same structural signature.
D0 is the atomic curriculum. D1 is the procedural matrix that holds the eval languages out. Repeat counts in that bank stay at or below 3, so four and five repetitions remain an extrapolation test.
The gate is a held-out composition mean above 0.90, a worst mixture above 0.85, repeat depths 4–5 above 0.90, atomic operators still near solved, and renaming invariance still about 1. Meeting it is the point at which a decoder would come back.

| Arch | Recipe | Seed | Params | Composition mean | Worst | Repeat 4–5 | Atomic min | Invariance | Gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h0 | d0 | 0 | 945464 | 0.748 | prefix_repeat 0.548 | 0.935 | 0.996 | 1.000 | near the atomic H0 mean |
| h0 | d1 | 0 | 945464 | 0.812 | repeat_optional 0.719 | 0.849 | 0.914 | 1.000 | above the atomic H0 mean, short of 0.82 |
| h0 | d1 | 1 | 945464 | 0.811 | repeat_optional 0.715 | 0.829 | 0.928 | 1.000 | above the atomic H0 mean, short of 0.82 |
| h0 | d1 | 2 | 945464 | 0.822 | repeat_optional 0.699 | 0.840 | 0.909 | 1.000 | partial composition gain, extrapolation under 0.90 |
| h1 | d0 | 0 | 1363558 | 0.759 | prefix_repeat 0.494 | 0.985 | 0.999 | 1.000 | near the atomic H0 mean |
| h1 | d1 | 0 | 1363558 | 0.793 | repeat_optional 0.699 | 0.888 | 0.939 | 1.000 | above the atomic H0 mean, short of 0.82 |
| h2 | d0 | 0 | 1311398 | 0.780 | prefix_repeat 0.602 | 0.977 | 0.999 | 1.000 | above the atomic H0 mean, short of 0.82 |
| h2 | d1 | 0 | 1311398 | 0.832 | repeat_optional 0.724 | 0.873 | 0.932 | 1.000 | partial composition gain, extrapolation under 0.90 |

## Slices

### h0 d0 seed 0

| Slice | Balanced | AUROC | Member acceptance | Invariance |
| --- | --- | --- | --- | --- |
| primitive | 0.988 | 0.996 | 1.000 | 1.000 |
| repeat | 0.975 | 0.999 | 0.963 | 1.000 |
| optional | 1.000 | 1.000 | 1.000 | 1.000 |
| alternation | 1.000 | 1.000 | 1.000 | 1.000 |
| repeat_extra | 0.775 | 0.935 | 0.613 | 1.000 |
| optional_internal | 0.750 | 1.000 | 0.500 | 1.000 |
| prefix_repeat | 0.469 | 0.548 | 0.188 | 1.000 |
| repeat_optional | 0.669 | 0.710 | 0.350 | 1.000 |
| prefix_alt | 0.844 | 0.987 | 0.700 | 1.000 |

### h0 d1 seed 0

| Slice | Balanced | AUROC | Member acceptance | Invariance |
| --- | --- | --- | --- | --- |
| primitive | 0.975 | 0.997 | 1.000 | 1.000 |
| repeat | 0.825 | 0.914 | 0.988 | 1.000 |
| optional | 1.000 | 1.000 | 1.000 | 1.000 |
| alternation | 0.925 | 0.991 | 0.975 | 1.000 |
| repeat_extra | 0.806 | 0.849 | 0.950 | 1.000 |
| optional_internal | 0.950 | 1.000 | 0.900 | 1.000 |
| prefix_repeat | 0.725 | 0.766 | 0.475 | 1.000 |
| repeat_optional | 0.669 | 0.719 | 0.350 | 1.000 |
| prefix_alt | 0.831 | 0.952 | 0.750 | 1.000 |

### h1 d0 seed 0

| Slice | Balanced | AUROC | Member acceptance | Invariance |
| --- | --- | --- | --- | --- |
| primitive | 0.988 | 0.999 | 1.000 | 1.000 |
| repeat | 0.906 | 1.000 | 1.000 | 1.000 |
| optional | 1.000 | 1.000 | 1.000 | 1.000 |
| alternation | 0.994 | 1.000 | 1.000 | 1.000 |
| repeat_extra | 0.831 | 0.985 | 1.000 | 1.000 |
| optional_internal | 0.750 | 0.972 | 0.500 | 1.000 |
| prefix_repeat | 0.381 | 0.494 | 0.163 | 1.000 |
| repeat_optional | 0.731 | 0.794 | 0.550 | 1.000 |
| prefix_alt | 0.669 | 0.989 | 0.350 | 1.000 |

### h1 d1 seed 0

| Slice | Balanced | AUROC | Member acceptance | Invariance |
| --- | --- | --- | --- | --- |
| primitive | 0.975 | 0.997 | 1.000 | 1.000 |
| repeat | 0.794 | 0.939 | 0.650 | 1.000 |
| optional | 0.988 | 1.000 | 0.975 | 1.000 |
| alternation | 0.869 | 0.982 | 1.000 | 1.000 |
| repeat_extra | 0.675 | 0.888 | 0.400 | 1.000 |
| optional_internal | 0.906 | 1.000 | 0.812 | 1.000 |
| prefix_repeat | 0.738 | 0.780 | 0.512 | 1.000 |
| repeat_optional | 0.562 | 0.699 | 0.138 | 1.000 |
| prefix_alt | 0.850 | 0.900 | 0.713 | 1.000 |

### h2 d0 seed 0

| Slice | Balanced | AUROC | Member acceptance | Invariance |
| --- | --- | --- | --- | --- |
| primitive | 0.988 | 0.999 | 1.000 | 1.000 |
| repeat | 0.988 | 1.000 | 0.988 | 1.000 |
| optional | 1.000 | 1.000 | 1.000 | 1.000 |
| alternation | 1.000 | 1.000 | 1.000 | 1.000 |
| repeat_extra | 0.875 | 0.977 | 0.800 | 1.000 |
| optional_internal | 0.750 | 0.965 | 0.500 | 1.000 |
| prefix_repeat | 0.519 | 0.602 | 0.087 | 1.000 |
| repeat_optional | 0.731 | 0.754 | 0.475 | 1.000 |
| prefix_alt | 0.887 | 0.985 | 0.787 | 1.000 |

### h2 d1 seed 0

| Slice | Balanced | AUROC | Member acceptance | Invariance |
| --- | --- | --- | --- | --- |
| primitive | 0.975 | 0.998 | 1.000 | 1.000 |
| repeat | 0.819 | 0.932 | 1.000 | 1.000 |
| optional | 1.000 | 1.000 | 1.000 | 1.000 |
| alternation | 0.938 | 0.994 | 1.000 | 1.000 |
| repeat_extra | 0.812 | 0.873 | 1.000 | 1.000 |
| optional_internal | 0.994 | 1.000 | 0.988 | 1.000 |
| prefix_repeat | 0.719 | 0.816 | 0.562 | 1.000 |
| repeat_optional | 0.681 | 0.724 | 0.487 | 1.000 |
| prefix_alt | 0.744 | 0.957 | 0.550 | 1.000 |

## Reading

The gate asks for a held-out composition mean above 0.90, a worst mixture above 0.85, repeat depths 4–5 above 0.90, atomic operators still near solved, and renaming invariance still about 1. One seed of H1 and one seed of H2 were trained on D0 and on D1. D0 isolates the architecture on the atomic curriculum. D1 is the procedural composition bank that already lifted the span model, with the eval languages held out.

H0 D0 seed 0: composition 0.748, worst prefix_repeat 0.548, repeat 4–5 0.935, atomic min 0.996, invariance 1.000, 945464 parameters.
H0 D1 over 3 seeds: composition 0.811–0.822, worst slice 0.699–0.719, repeat 4–5 0.829–0.849.

H1 D0 seed 0: composition 0.759, worst prefix_repeat 0.494, repeat 4–5 0.985, atomic min 0.999, invariance 1.000, 1363558 parameters. Near the atomic H0 mean.
H1 D1 seed 0: composition 0.793, worst repeat_optional 0.699, repeat 4–5 0.888, atomic min 0.939, invariance 1.000, 1363558 parameters. Above the atomic H0 mean, short of 0.82.
H2 D0 seed 0: composition 0.780, worst prefix_repeat 0.602, repeat 4–5 0.977, atomic min 0.999, invariance 1.000, 1311398 parameters. Above the atomic H0 mean, short of 0.82.
H2 D1 seed 0: composition 0.832, worst repeat_optional 0.724, repeat 4–5 0.873, atomic min 0.932, invariance 1.000, 1311398 parameters. Partial composition gain, extrapolation under 0.90.

H2 is the stronger mixer. On the atomic curriculum it moves prefix+repeat up from the H0 value of 0.548 and keeps repeat depths 4–5 above 0.90. On D1 its composition mean sits just above the H0 D1 range, and the worst mixture remains repeat+optional. H1 on D1 stays under the H0 D1 range. Invariance stays 1.000 on every hierarchical seed, so the rename symmetry holds for every run. The gain is a few points of AUROC. Explicit group relations preserve that symmetry and, on the atomic curriculum, the depth test. The operator algebra the gate asked for is still ahead of both models. The decoder stays off.

