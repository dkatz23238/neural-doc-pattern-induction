# Membership backbone

Stage 1 asks whether a small model can represent the language behind a few strings.
The input is the support set and one query. The output is the probability that the query belongs.
Training starts on prefixes and classes, then delimited fields, then repeats and alternations, then the full schema mixture. Those schemas never contain `#`, `~`, `*`, `=`, or `§`. The selection split uses those bytes, longer classes, and the harder constructions.
Bytes keep a frozen identity code, so an unseen delimiter still matches itself, and a pointer copies the aligned support state onto the query.
An auxiliary head marks literal and delimiter bytes on positive queries. There is no regex decoder in this run.

Trainable parameters: 1877186. Best epoch: 6.
The checkpoint is the one with the highest shifted balanced accuracy. BENCH-C was not used to stop.

## Shifted holdout

| System | Accuracy | Balanced | TPR | TNR | N |
| --- | --- | --- | --- | --- | --- |
| membership model | 0.556 | 0.556 | 0.444 | 0.669 | 640, AUROC 0.590 |
| structural heuristic | 0.716 | 0.716 | 0.884 | 0.547 | 640 |
| code-length rule | 0.820 | 0.820 | 0.866 | 0.775 | 640 |

Balanced accuracy is (TPR + TNR) / 2, the same shape as semantic accuracy.

On the 332 queries that actually contain a held-out byte, the model balanced score is 0.521 (AUROC 0.571) and the heuristic is 0.660.

Where the heuristic is wrong (182 queries), the model is right 60.4% of the time. Where the heuristic is right (458 queries), the model stays right 53.7% of the time.

The same comparison against the code-length rule: recovery 50.4% of 115 errors, preservation 56.8%.

## Few-shot and in-distribution

After the prefix-and-class stage, fresh languages in the training alphabet score 0.844 balanced (AUROC 0.901). That is the same model family on an easier slice, and it is the part that actually trains.

Shifted languages shown with only 2 supports: model 0.519, heuristic 0.509, rule 0.500.

In-distribution full schemas, not used for selection: model 0.632, heuristic 0.774.

## BENCH-C probe

Untouched document tasks, scored once, not used to stop. Queries longer than 48 bytes were dropped (240). Held-out characters in this probe are #*=~§.

| System | Accuracy | Balanced | TPR | TNR | N |
| --- | --- | --- | --- | --- | --- |
| membership model | 0.508 | 0.508 | 0.550 | 0.466 | 26352, AUROC 0.495 |
| structural heuristic | 0.935 | 0.935 | 0.921 | 0.949 | 26352 |
| code-length rule | 0.958 | 0.958 | 0.928 | 0.987 | 26352 |

## Reading

The network learns the primitive task and does not transfer it. Prefix-and-class languages in the training alphabet reach 0.844 balanced (AUROC 0.901). On the shifted split the selected checkpoint scores 0.556 balanced (AUROC 0.590) against 0.716 for the structural heuristic and 0.820 for the code-length rule. Where a query actually contains `#`, `~`, `*`, `=`, or `§`, balanced accuracy is 0.521 and true members are accepted only 24.7% of the time: an unfamiliar byte is treated as evidence against membership, rather than as a delimiter that is constant across the support. In-distribution full schemas reach 0.632 against the heuristic's 0.774. BENCH-C, scored once, is 0.508 against 0.935. The model is right on 60.4% of the heuristic's mistakes and on 53.7% of the heuristic's successes, so it is not a residual corrector either. Stage 1 does not yet show a language representation that a regex decoder can sit on.

Epochs

- epoch 1 (prefix-class): loss 1.014, bce 0.626, simple 0.844, shifted balanced 0.528, in-distribution balanced 0.533
- epoch 2 (delimited): loss 0.866, bce 0.587, simple 0.681, shifted balanced 0.533, in-distribution balanced 0.514
- epoch 3 (repeat-alt): loss 0.844, bce 0.566, simple 0.831, shifted balanced 0.533, in-distribution balanced 0.542
- epoch 4 (full): loss 1.080, bce 0.685, simple 0.719, shifted balanced 0.544, in-distribution balanced 0.594
- epoch 5 (full): loss 1.040, bce 0.672, simple 0.719, shifted balanced 0.552, in-distribution balanced 0.580
- epoch 6 (full): loss 1.005, bce 0.658, simple 0.738, shifted balanced 0.556, in-distribution balanced 0.632
- epoch 7 (full): loss 0.980, bce 0.647, simple 0.694, shifted balanced 0.548, in-distribution balanced 0.642
