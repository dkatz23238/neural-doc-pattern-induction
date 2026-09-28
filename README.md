# Document pattern induction

Experiments for a go/no-go decision: can a small local model infer a useful regular expression from a handful of example strings, or do symbolic baselines already close the gap?

The score that matters is semantic accuracy on a hidden set, not similarity of the regex text.

```text
semantic accuracy = (hidden positive recall + hidden negative rejection) / 2
```

`.*` and an alternation of the input strings both score poorly. The first accepts the negatives. The second misses the hidden positives.

## Layout

| Path | Role |
| --- | --- |
| `pattern_induction/evaluate.py` | Compile a regex and score it on hidden positives and negatives |
| `pattern_induction/heuristic.py` | B1 structural heuristic |
| `pattern_induction/baselines.py` | Memorization, grex configurations, fAST |
| `pattern_induction/bench_c.py` | Curated document tasks. Never used as training data |
| `pattern_induction/bench_b.py` | Document-style synthetic tasks |
| `pattern_induction/bench_a.py` | Controlled AST tasks with a shape holdout |
| `pattern_induction/bench_d.py` | Same schemas shown with 2, 3, 5, 10, or 20 examples |
| `pattern_induction/frontier.py` | Prompts for the two frontier conditions. No call without an API key |
| `results/exp1/` | First symbolic run and the Gate 1 HTML report |
| `results/exp1/exp1_report.html` | Self-contained findings page |

## Run

```bash
pip install -r requirements.txt
cargo install grex
# fAST optionally renders Graphviz; install the `graphviz` package or leave the display hook disabled
export PYTHONPATH=.
python scripts/generate_benchmarks.py
python scripts/run_experiment1.py
python -m unittest tests.test_core
```

`generate_benchmarks.py` writes a review-sized suite (about 550 BENCH-C tasks, 1,500 BENCH-B, 320 BENCH-D, 1,200 BENCH-A). The BENCH-A generator is the same function used for the 100k/10k/10k corpus; pass `--bench-a-dev 100000 --bench-a-val 10000 --bench-a-test 10000` for the full split. Shape buckets are hashed, so the larger draw keeps the same held-out shapes.

The candidate-oracle run does not train a neural net. It builds a small symbolic neighborhood of the heuristic, scores the best candidate in that neighborhood on the hidden set, and compares that ceiling with two label-free scorers and a ridge regression on visible features.

```bash
PYTHONPATH=. python scripts/run_candidate_oracle.py
```

`results/exp2/candidate_oracle.md` is the written result. BENCH-C is not used to fit the linear models.

The CPU ranker pilot trains the tiny cross-attention model and a pooled ablation. Structural models train on BENCH-A dev and stop on BENCH-A val semantic accuracy. A document model trains on the BENCH-B fold. BENCH-C is scored once and is not used to stop.

```bash
PYTHONPATH=. python scripts/run_ranker_pilot.py
```

`results/exp2/ranker_pilot.md` records train loss, validation semantic accuracy, and the held-out comparison with the code-length rule. On this CPU run neither the shape-trained model nor the document-trained model beats that rule by a point on its decision split, and cross-attention stays within half a point of the pooled model on BENCH-A test.

The structural-generator run asks a different question: can a small pool contain the right kind of regex, rather than only mutations of the heuristic. It adds span merging, delimiter and repeat structure, optional spans, sentinel alternation, and a short code-length beam.

```bash
PYTHONPATH=. python scripts/run_structural_oracle.py
```

`results/exp2/structural_oracle.md` reports oracle@8, oracle@16, oracle@32, and the code-length rule on that pool. BENCH-C is scored only.

The selector run freezes that pool at 16 candidates and trains a listwise linear scorer on visible features. Labels are hidden semantic accuracy on the BENCH-B hash fold. Temperature, L2, and a confidence margin back to the old 8-candidate code-length rule are chosen on the BENCH-B validation fold. BENCH-C is scored once.

```bash
PYTHONPATH=. python scripts/run_selector.py
```

`results/exp2/selector.md` is the written result.

The residual-override run asks a narrower question: starting from that old rule, when should a different candidate in the frozen 16 replace it? The target is the hidden-accuracy gap. False overrides are weighted more heavily than missed gains. Training and validation use disjoint document formats, and BENCH-C is scored once.

```bash
PYTHONPATH=. python scripts/run_override.py
```

`results/exp2/override.md` is the written result.

The latent-schema run stops ranking candidates. A tiny set encoder predicts literal spans, classes, repeats, optional tails, and alternations, and a compiler turns that schema into a regex. Epochs stop on a shifted synthetic split. BENCH-C is scored once.

```bash
PYTHONPATH=. python scripts/run_schema.py
```

`results/exp2/schema.md` is the written result.

The membership run is the first stage of a set-conditioned inducer. A token-level model predicts whether a new string belongs to the language of a few examples. Literal identity uses a frozen byte code and a pointer, not a closed output vocabulary. Training and the stopping split use disjoint delimiter bytes. BENCH-C is scored once.

```bash
PYTHONPATH=. python scripts/run_membership.py
```

`results/exp2/membership.md` is the written result.

The renaming run asks whether that membership model is using symbol identity. It renames letters, digits, and separators inside a task, including sending `-` to `§`, and scores the probability change. A second encoder replaces raw codepoints with structural class, a task-local symbol id, and alignment relations. BENCH-C is not scored.

```bash
PYTHONPATH=. python scripts/run_renaming.py
```

`results/exp2/renaming.md` is the written result.

The composition run freezes those relational features and asks whether the model can learn repetition, optionality, and alternation. A span encoder groups class runs. The generator's own groups are auxiliary labels. Training stays on the atomic operators. Repeats of length 4–5, an optional group in the middle, and mixtures such as prefix+repeat are scored only at the end. BENCH-C is not scored.

```bash
PYTHONPATH=. python scripts/run_compose.py
```

`results/exp2/compose.md` is the written result.

The data-recipe run freezes that model. Same parameter band, optimizer, 1200 steps, batch, loss, and evaluation slices. D0 repeats the atomic curriculum. D1 adds other compositions and holds the eval languages out. D2 and D3 train on schemas proposed by `gpt-5.6-terra` and accepted by the compiler. D4 adds templates aimed at the remaining failures. Teacher calls are cached under `teacher_runs/` and `data/teacher/`. The spend cap in code is $12.

```bash
PYTHONPATH=. python scripts/run_data_recipes.py --stage teacher --prompt composition_v1.md --kind schemas
PYTHONPATH=. python scripts/run_data_recipes.py --stage train --recipe d0
```

`results/data_recipe/summary.md` is the written result.

The hierarchy run keeps those recipes and that training budget. H0 is the span model above. H1 adds group pooling and a group transformer. H2 replaces the group mixer with typed parent, sibling, repeat-unit, and alignment edges. The teacher cache is reused and is not called again.

```bash
PYTHONPATH=. python scripts/run_hierarchy.py --stage train --arch h2 --recipe d0
```

`results/hierarchy/summary.md` is the written result.

The operator-algebra run keeps those recipes and that budget. H3-oracle folds one shared sequence, repeat, optional, and alternation module over the generator's tree. The query annotation is not an input. Repeat bounds come from the schema, so depths 4–5 stay an extrapolation test. The teacher is not called. Structure induction waits on this oracle.

```bash
PYTHONPATH=. python scripts/run_operators.py --stage train --arch h3o --recipe d0
```

`results/operator_algebra/summary.md` is the written result.

Grex configs are `grex_default`, `grex_digits` (`-d`), `grex_words` (`-w`), `grex_repetitions` (`-r`), and `grex_combined` (`-d -w -r`).

fAST is the ICGI 2023 reference implementation (`find-abstract-syntax-tree`). It searches character by character and often returns nothing within a few seconds on document-length strings. The runner records timeouts instead of inventing a regex. Its concatenation operator is translated into a Python regex before scoring.

## Frontier models

Gate 1 scores two frontier models on BENCH-C with the simple prompt. The pair called here is `gpt-6-luna` and `gpt-5.6-terra` (`gpt-6-terra` is not available on this key). Neither model accepts temperature 0, so the Responses API calls omit it. `results/exp1/frontier_bench_c.json` records the model id, date, and task count. `results/exp1/frontier_prompts_sample.json` stores the simple, guarded, and structural-feature prompts.

## What this run does not do

Gate 1 does not train a local model. On the 2026-09-25 BENCH-C run, `gpt-6-luna` scored 95.5% and `gpt-5.6-terra` scored 98.1% semantic accuracy, against a 93.2% heuristic. The gaps are +2.3 and +4.9 points, short of the 8–10 point rule, so Gate 1 is a no. Macro accuracy (mean of per-domain scores) is reported beside the micro average so easy invoice-number columns do not decide the gate by themselves.

BENCH-C is a curated catalog of document fields with a human-specified gold regex for each column. It is not a scrape of private PDFs. Hidden negatives are near-miss mutations that fall outside that regex.
