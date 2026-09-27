# Frozen-pool selector

The structural generator stays as it is. Each task offers 16 candidates, the prefix of the structural round-robin. A linear scorer is trained listwise: the target is a softmax over hidden semantic accuracy, and the inputs are visible features only. If the top two probabilities are close, the scorer keeps the code-length choice from the original 8-candidate neighborhood.

Training tasks are the BENCH-B hash fold. Temperature, L2, and the margin are chosen on the held-out fifth of BENCH-B. BENCH-C is scored once. BENCH-A is not used.

## Reading

On the frozen 16-candidate pool, BENCH-C oracle is 99.5%. The old code-length rule scores 96.2% (regressions 0.7%). Listwise linear scores 96.9% (regressions 7.4%). The confidence-gated scorer scores 96.9% (regressions 7.4%). Charging alternation literals by hand scores 95.3% (regressions 1.8%).

The chosen margin is 0.0. On the BENCH-B validation fold, margin 0 regresses 1.0% of tasks and scores 98.5%, so it stays inside the 2% budget and wins. Every margin of 0.05 or wider scores 95.9% there, matching the old rule. The gate does not produce a setting between trusting the listwise model on every task and falling back on every task.

The gain sits on the heuristic's too-broad failures (old rule 67.1%, listwise 96.2%, oracle 99.6%, n=10). The success majority moves the other way (heuristic 99.1%, listwise 98.2%, n=402). Variable-length stays far from the ceiling (listwise 84.6%, oracle 99.7%, n=34).

This misses the gate of about 98% BENCH-C with regressions under 2%. The cheap scorer leaves 2.6 points versus oracle@16, and it regresses more tasks than the old rule. A tiny neural ranker trained listwise on the same frozen pool is the next selector, not a larger candidate generator. It still has to keep the regression rate near the old rule's.

BENCH-D was not used for selection. There the listwise scorer scores 91.7%, the old rule 93.8%, and oracle@16 97.8%.

BENCH-B and BENCH-C share document field families, so this BENCH-C number is an upper bound on transfer. Temperature, L2, and the margin were frozen on the BENCH-B validation fold before BENCH-C was scored.

## Protocol

Listwise temperature 0.05, L2 0.01, stopped at step 1 with validation accuracy 98.5%. Pointwise ridge lambda 0.1 (validation 96.3%). Confidence margin 0.0. Features: 28.

## B-val

Validation fold. Used to choose temperature, L2, and the margin.

| System | Semantic accuracy | Macro | Improved | Regressed | Unchanged |
| --- | --- | --- | --- | --- | --- |
| heuristic | 91.9% | 90.6% | 0.0% | 0.0% | 100.0% |
| code-length rule, old 8 | 95.9% | 95.6% | 19.9% | 0.3% | 79.8% |
| code-length rule, structural 16 | 95.5% | 94.7% | 18.2% | 1.3% | 80.5% |
| code length plus alternation-literal charge | 96.2% | 95.6% | 19.5% | 0.3% | 80.1% |
| pointwise ridge on hidden accuracy | 96.3% | 95.9% | 20.9% | 11.9% | 67.2% |
| listwise linear | 98.5% | 98.7% | 21.2% | 1.0% | 77.8% |
| listwise, else the old rule | 98.5% | 98.7% | 21.2% | 1.0% | 77.8% |
| oracle@16 | 99.6% | 99.6% | 30.5% | 0.0% | 69.5% |

## C

Untouched document tasks. Families overlap BENCH-B, so this is an upper bound on transfer.

| System | Semantic accuracy | Macro | Improved | Regressed | Unchanged |
| --- | --- | --- | --- | --- | --- |
| heuristic | 93.2% | 92.1% | 0.0% | 0.0% | 100.0% |
| code-length rule, old 8 | 96.2% | 95.7% | 14.4% | 0.7% | 84.8% |
| code-length rule, structural 16 | 95.4% | 94.9% | 20.2% | 6.0% | 73.8% |
| code length plus alternation-literal charge | 95.3% | 94.8% | 15.0% | 1.8% | 83.2% |
| pointwise ridge on hidden accuracy | 95.4% | 94.5% | 20.9% | 25.8% | 53.2% |
| listwise linear | 96.9% | 96.5% | 20.4% | 7.4% | 72.2% |
| listwise, else the old rule | 96.9% | 96.5% | 20.4% | 7.4% | 72.2% |
| oracle@16 | 99.5% | 99.5% | 39.2% | 0.0% | 60.8% |

Heuristic-failure families:

| Failure | n | heuristic | old rule | listwise | gated | oracle@16 |
| --- | --- | --- | --- | --- | --- | --- |
| success | 402 | 99.1% | 99.1% | 98.2% | 98.2% | 99.7% |
| optional region | 50 | 85.3% | 88.7% | 92.3% | 92.3% | 97.8% |
| variable-length inference | 34 | 81.4% | 82.0% | 84.6% | 84.6% | 99.7% |
| character-class inference | 27 | 80.6% | 98.2% | 99.5% | 99.5% | 99.5% |
| too narrow | 22 | 60.6% | 98.8% | 100.0% | 100.0% | 100.0% |
| too_broad | 10 | 60.8% | 67.1% | 96.2% | 96.2% | 99.6% |
| mixed population | 4 | 63.5% | 91.1% | 100.0% | 100.0% | 100.0% |
| too broad | 3 | 68.1% | 70.8% | 89.6% | 89.6% | 100.0% |
| OCR noise | 1 | 79.2% | 100.0% | 100.0% | 100.0% | 100.0% |
| alternation | 1 | 89.6% | 89.6% | 89.6% | 89.6% | 100.0% |

## D

Sample-size curve. Not used to fit or to choose the margin.

| System | Semantic accuracy | Macro | Improved | Regressed | Unchanged |
| --- | --- | --- | --- | --- | --- |
| heuristic | 89.8% | 89.8% | 0.0% | 0.0% | 100.0% |
| code-length rule, old 8 | 93.8% | 93.8% | 16.6% | 0.0% | 83.4% |
| code-length rule, structural 16 | 90.7% | 90.7% | 19.4% | 12.5% | 68.1% |
| code length plus alternation-literal charge | 93.0% | 93.0% | 14.7% | 3.1% | 82.2% |
| pointwise ridge on hidden accuracy | 91.1% | 91.1% | 19.1% | 48.8% | 32.2% |
| listwise linear | 91.7% | 91.7% | 18.8% | 9.4% | 71.9% |
| listwise, else the old rule | 91.7% | 91.7% | 18.8% | 9.4% | 71.9% |
| oracle@16 | 97.8% | 97.8% | 40.3% | 0.0% | 59.7% |
