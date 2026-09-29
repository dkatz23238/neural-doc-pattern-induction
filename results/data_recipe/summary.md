# Data recipes for compositional generalization

The span model, the loss, the step count, and the evaluation slices are the ones from the compositional run. The training distribution is the only change.
D0 is the atomic curriculum. D1 adds a procedural matrix of other compositions and holds the eval languages out. D2 and D3 train on schemas a teacher proposed and the compiler accepted. D4 adds templates aimed at the remaining failures.
D3 and D4 may include the eval families as contrasts. D0, D1, and D2 do not. Repeat counts in teacher schemas stay at or below 3, so four and five repetitions remain an extrapolation test.
Held-out composition AUROC is the mean of prefix+repeat, repeat+optional, and prefix+alternation. The worst of those three is reported beside it. Renaming invariance below 0.99 rejects a recipe.

| Recipe | Seed | Composition mean | Worst | Repeat 4–5 | Invariance | Exposes eval family | Band |
| --- | --- | --- | --- | --- | --- | --- | --- |
| d0 | 0 | 0.748 | prefix_repeat 0.548 | 0.935 | 1.000 | [] | near the atomic baseline |
| d1 | 0 | 0.812 | repeat_optional 0.719 | 0.849 | 1.000 | [] | above the atomic baseline, short of 0.82 |
| d1 | 1 | 0.811 | repeat_optional 0.715 | 0.829 | 1.000 | [] | above the atomic baseline, short of 0.82 |
| d1 | 2 | 0.822 | repeat_optional 0.699 | 0.840 | 1.000 | [] | mean in the 0.82–0.87 band; worst slice under 0.85 |
| d2 | 0 | 0.816 | prefix_repeat 0.735 | 0.826 | 1.000 | [] | above the atomic baseline, short of 0.82 |
| d2 | 1 | 0.861 | repeat_optional 0.803 | 0.826 | 1.000 | [] | mean in the 0.82–0.87 band; worst slice under 0.85 |
| d2 | 2 | 0.803 | prefix_repeat 0.734 | 0.834 | 1.000 | [] | above the atomic baseline, short of 0.82 |
| d3 | 0 | 0.801 | repeat_optional 0.752 | 0.839 | 1.000 | ['repeat_optional'] | above the atomic baseline, short of 0.82 |
| d4 | 0 | 0.786 | repeat_optional 0.630 | 0.848 | 1.000 | ['prefix_alt', 'prefix_repeat', 'repeat_optional'] | above the atomic baseline, short of 0.82 |

## Slices

### d0 seed 0

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

### d1 seed 0

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

### d1 seed 1

| Slice | Balanced | AUROC | Member acceptance | Invariance |
| --- | --- | --- | --- | --- |
| primitive | 0.975 | 0.999 | 1.000 | 1.000 |
| repeat | 0.819 | 0.928 | 0.925 | 1.000 |
| optional | 0.988 | 1.000 | 1.000 | 1.000 |
| alternation | 0.887 | 0.993 | 1.000 | 1.000 |
| repeat_extra | 0.681 | 0.829 | 0.550 | 1.000 |
| optional_internal | 1.000 | 1.000 | 1.000 | 1.000 |
| prefix_repeat | 0.725 | 0.754 | 0.475 | 1.000 |
| repeat_optional | 0.694 | 0.715 | 0.463 | 1.000 |
| prefix_alt | 0.869 | 0.964 | 0.800 | 1.000 |

### d1 seed 2

| Slice | Balanced | AUROC | Member acceptance | Invariance |
| --- | --- | --- | --- | --- |
| primitive | 0.938 | 0.996 | 0.887 | 1.000 |
| repeat | 0.850 | 0.909 | 0.988 | 1.000 |
| optional | 1.000 | 1.000 | 1.000 | 1.000 |
| alternation | 0.906 | 1.000 | 1.000 | 1.000 |
| repeat_extra | 0.806 | 0.840 | 0.963 | 1.000 |
| optional_internal | 0.756 | 1.000 | 0.512 | 1.000 |
| prefix_repeat | 0.581 | 0.794 | 0.163 | 1.000 |
| repeat_optional | 0.588 | 0.699 | 0.175 | 1.000 |
| prefix_alt | 0.887 | 0.974 | 0.812 | 1.000 |

### d2 seed 0

| Slice | Balanced | AUROC | Member acceptance | Invariance |
| --- | --- | --- | --- | --- |
| primitive | 0.975 | 0.998 | 1.000 | 1.000 |
| repeat | 0.831 | 0.938 | 1.000 | 1.000 |
| optional | 1.000 | 1.000 | 1.000 | 1.000 |
| alternation | 0.919 | 0.990 | 1.000 | 1.000 |
| repeat_extra | 0.819 | 0.826 | 1.000 | 1.000 |
| optional_internal | 0.769 | 0.997 | 0.537 | 1.000 |
| prefix_repeat | 0.644 | 0.735 | 0.400 | 1.000 |
| repeat_optional | 0.700 | 0.803 | 0.838 | 1.000 |
| prefix_alt | 0.744 | 0.909 | 0.550 | 1.000 |

### d2 seed 1

| Slice | Balanced | AUROC | Member acceptance | Invariance |
| --- | --- | --- | --- | --- |
| primitive | 0.975 | 0.996 | 1.000 | 1.000 |
| repeat | 0.806 | 0.921 | 0.900 | 1.000 |
| optional | 1.000 | 1.000 | 1.000 | 1.000 |
| alternation | 0.894 | 0.987 | 0.800 | 1.000 |
| repeat_extra | 0.762 | 0.826 | 0.850 | 1.000 |
| optional_internal | 0.650 | 0.963 | 0.300 | 1.000 |
| prefix_repeat | 0.544 | 0.803 | 0.087 | 1.000 |
| repeat_optional | 0.675 | 0.803 | 0.388 | 1.000 |
| prefix_alt | 0.831 | 0.977 | 0.688 | 1.000 |

### d2 seed 2

| Slice | Balanced | AUROC | Member acceptance | Invariance |
| --- | --- | --- | --- | --- |
| primitive | 0.956 | 0.993 | 1.000 | 1.000 |
| repeat | 0.825 | 0.916 | 0.975 | 1.000 |
| optional | 1.000 | 1.000 | 1.000 | 1.000 |
| alternation | 0.869 | 0.954 | 0.887 | 1.000 |
| repeat_extra | 0.781 | 0.834 | 0.925 | 1.000 |
| optional_internal | 0.869 | 1.000 | 0.738 | 1.000 |
| prefix_repeat | 0.725 | 0.734 | 0.550 | 1.000 |
| repeat_optional | 0.694 | 0.798 | 0.725 | 1.000 |
| prefix_alt | 0.762 | 0.878 | 0.900 | 1.000 |

### d3 seed 0

| Slice | Balanced | AUROC | Member acceptance | Invariance |
| --- | --- | --- | --- | --- |
| primitive | 0.981 | 1.000 | 1.000 | 1.000 |
| repeat | 0.812 | 0.925 | 0.787 | 1.000 |
| optional | 0.925 | 1.000 | 1.000 | 1.000 |
| alternation | 0.988 | 1.000 | 0.988 | 1.000 |
| repeat_extra | 0.606 | 0.839 | 0.275 | 1.000 |
| optional_internal | 0.994 | 1.000 | 0.988 | 1.000 |
| prefix_repeat | 0.656 | 0.814 | 0.325 | 1.000 |
| repeat_optional | 0.606 | 0.752 | 0.212 | 1.000 |
| prefix_alt | 0.825 | 0.837 | 1.000 | 1.000 |

### d4 seed 0

| Slice | Balanced | AUROC | Member acceptance | Invariance |
| --- | --- | --- | --- | --- |
| primitive | 0.969 | 0.998 | 1.000 | 1.000 |
| repeat | 0.825 | 0.897 | 0.963 | 1.000 |
| optional | 0.912 | 1.000 | 1.000 | 1.000 |
| alternation | 0.819 | 0.962 | 0.738 | 1.000 |
| repeat_extra | 0.769 | 0.848 | 0.850 | 1.000 |
| optional_internal | 0.944 | 1.000 | 1.000 | 1.000 |
| prefix_repeat | 0.725 | 0.831 | 0.762 | 1.000 |
| repeat_optional | 0.550 | 0.630 | 0.988 | 1.000 |
| prefix_alt | 0.800 | 0.898 | 0.938 | 1.000 |

## Reading

The success bar for this phase is a held-out composition mean above 0.90 and a worst mixture above 0.85, with renaming invariance still about 1. A mean from 0.82 to 0.87 is evidence that the curriculum matters. A mean near 0.75 means the data variable did not move the model. A mean above 0.93 with repeat-depth extrapolation still strong would be the point to attach a decoder. Invariance under 0.99 rejects the recipe.

d0 composition mean 0.748–0.748 over 1 seed, worst slice 0.548–0.548, repeat 4–5 AUROC 0.935–0.935. Eval families in that bank: none.
d1 composition mean 0.811–0.822 over 3 seeds, worst slice 0.699–0.719, repeat 4–5 AUROC 0.829–0.849. Eval families in that bank: none.
d2 composition mean 0.803–0.861 over 3 seeds, worst slice 0.734–0.803, repeat 4–5 AUROC 0.826–0.834. Eval families in that bank: none.
d3 composition mean 0.801–0.801 over 1 seed, worst slice 0.752–0.752, repeat 4–5 AUROC 0.839–0.839. Eval families in that bank: ['repeat_optional'].
d4 composition mean 0.786–0.786 over 1 seed, worst slice 0.630–0.630, repeat 4–5 AUROC 0.848–0.848. Eval families in that bank: ['prefix_alt', 'prefix_repeat', 'repeat_optional'].

D1 and D2 keep the three eval mixtures out of the training bank. Their means sit around 0.81, and one D2 seed reaches the 0.82–0.87 band. The worst mixture stays below 0.85 on every seed. Prefix+repeat rises from the atomic 0.548 into the mid 0.70s or low 0.80s, and it is still the weak slice on several seeds. Repeat depths 4 and 5 drop from the atomic 0.935 into the low 0.80s, so the composition gain trades away some depth extrapolation.
D3 contrasts one structural attachment at a time. Its bank includes a repeat-then-optional language, and that eval slice does not become the strong one. D4 adds failure-targeted schemas, including prefix+repeat itself, and the mean does not rise. No recipe reaches the 0.90 / 0.85 bar. Invariance stays 1.000, so none are rejected for a broken rename. The decoder stays off.

Teacher spend recorded in the ledger: $0.1625 across 5 calls. Rates used for that figure are $2 / $12 per million input / output tokens for gpt-5.6-terra.

