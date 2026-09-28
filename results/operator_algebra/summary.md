# Operator algebra

H2 is the typed group-graph control. H3-oracle keeps that trunk, those recipes, the 1200-step budget, and the same evaluation slices. The change is the mixer: one shared function each for sequence, repeat, optional, and alternation, folded over the generator's operator tree.

The model is given that tree. Support strings still supply the leaf bindings. The query annotation stays out of the forward pass. Repeat bounds are the schema's min and max, so a query with four or five copies is the same repeat operator the support showed at two to four.

The oracle bar is a held-out composition mean above 0.95, because the structure is handed in. The product gate stays a mean above 0.90, a worst mixture above 0.85, repeat depths 4–5 above 0.90, and renaming invariance about 1. A mean near 0.80 means this operator algebra still needs a different design, and structure induction waits.

| Arch | Recipe | Seed | Params | Composition mean | Worst | Repeat 4–5 | Atomic min | Invariance | Reading |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h0 | d0 | 0 | 945464 | 0.748 | prefix_repeat 0.548 | 0.935 | 0.996 | 1.000 | control |
| h2 | d0 | 0 | 1311398 | 0.780 | prefix_repeat 0.602 | 0.977 | 0.999 | 1.000 | control |
| h3o | d0 | 0 | 1330081 | 0.769 | prefix_repeat 0.650 | 0.837 | 0.892 | 1.000 | gold tree did not make the operator algebra compose, and an atomic operator slipped under 0.90; do not build structure induction |
| h0 | d1 | 0 | 945464 | 0.812 | repeat_optional 0.719 | 0.849 | 0.914 | 1.000 | control |
| h2 | d1 | 0 | 1311398 | 0.832 | repeat_optional 0.724 | 0.873 | 0.932 | 1.000 | control |
| h3o | d1 | 0 | 1330081 | 0.784 | repeat_optional 0.635 | 0.873 | 0.894 | 1.000 | gold tree did not make the operator algebra compose, and an atomic operator slipped under 0.90; do not build structure induction |

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

### h3o d0 seed 0

| Slice | Balanced | AUROC | Member acceptance | Invariance |
| --- | --- | --- | --- | --- |
| primitive | 0.956 | 0.992 | 1.000 | 1.000 |
| repeat | 0.825 | 0.892 | 1.000 | 1.000 |
| optional | 0.988 | 1.000 | 1.000 | 1.000 |
| alternation | 0.994 | 1.000 | 1.000 | 1.000 |
| repeat_extra | 0.819 | 0.837 | 1.000 | 1.000 |
| optional_internal | 0.981 | 1.000 | 0.963 | 1.000 |
| prefix_repeat | 0.600 | 0.650 | 0.600 | 1.000 |
| repeat_optional | 0.506 | 0.766 | 0.013 | 1.000 |
| prefix_alt | 0.919 | 0.890 | 0.850 | 1.000 |

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

### h3o d1 seed 0

| Slice | Balanced | AUROC | Member acceptance | Invariance |
| --- | --- | --- | --- | --- |
| primitive | 0.969 | 0.988 | 1.000 | 1.000 |
| repeat | 0.812 | 0.894 | 0.963 | 1.000 |
| optional | 0.988 | 1.000 | 0.975 | 1.000 |
| alternation | 0.969 | 1.000 | 1.000 | 1.000 |
| repeat_extra | 0.806 | 0.873 | 0.900 | 1.000 |
| optional_internal | 0.956 | 0.998 | 0.988 | 1.000 |
| prefix_repeat | 0.681 | 0.731 | 0.588 | 1.000 |
| repeat_optional | 0.525 | 0.635 | 0.050 | 1.000 |
| prefix_alt | 0.975 | 0.988 | 0.963 | 1.000 |

## Reading

H3-oracle D0 seed 0: composition 0.769, worst prefix_repeat 0.650, repeat 4–5 0.837, atomic min 0.892, invariance 1.000, 1330081 parameters, selected step 1000. gold tree did not make the operator algebra compose, and an atomic operator slipped under 0.90; do not build structure induction.

H3-oracle D1 seed 0: composition 0.784, worst repeat_optional 0.635, repeat 4–5 0.873, atomic min 0.894, invariance 1.000, 1330081 parameters, selected step 1100. gold tree did not make the operator algebra compose, and an atomic operator slipped under 0.90; do not build structure induction.

On D0, H2 was composition 0.780, worst prefix+repeat 0.602, repeat 4–5 0.977. H3-oracle is 0.769 / 0.650 / 0.837. Prefix+repeat moves 0.602 → 0.650. Repeat+optional moves 0.754 → 0.766. Prefix+alternation moves 0.985 → 0.890. Atomic repeat moves 1.000 → 0.892. Optional and alternation stay solved. The depth test, which H2 held at 0.977, falls to 0.837. Repeat+optional member acceptance is 0.013: the optional node is in the gold tree, and almost no true member of that mixture crosses 0.5.

On D1, H2 was composition 0.832, worst repeat+optional 0.724, repeat 4–5 0.873. H3-oracle is 0.784 / 0.635 / 0.873. Prefix+repeat is 0.731 where H2 reached 0.816. The procedural bank leaves the same gap.

With the correct tree in hand, the shared modules stay near the H2 mean on D0 and below H2 on D1. The worst mixture stays under 0.85, and D0 repeat depth falls from 0.977 to 0.837. Renaming invariance stays 1.000, so the relational features are intact. H3-induced and latent-schema distillation wait. The decoder stays off. The open problem is the operator functions themselves.

