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

Grex configs are `grex_default`, `grex_digits` (`-d`), `grex_words` (`-w`), `grex_repetitions` (`-r`), and `grex_combined` (`-d -w -r`).

fAST is the ICGI 2023 reference implementation (`find-abstract-syntax-tree`). It searches character by character and often returns nothing within a few seconds on document-length strings. The runner records timeouts instead of inventing a regex. Its concatenation operator is translated into a Python regex before scoring.

## Frontier models

Gate 1 scores two frontier models on BENCH-C with the simple prompt. The pair called here is `gpt-6-luna` and `gpt-5.6-terra` (`gpt-6-terra` is not available on this key). Neither model accepts temperature 0, so the Responses API calls omit it. `results/exp1/frontier_bench_c.json` records the model id, date, and task count. `results/exp1/frontier_prompts_sample.json` stores the simple, guarded, and structural-feature prompts.

## What this run does not do

No neural model is trained. On the 2026-09-25 BENCH-C run, `gpt-6-luna` scored 95.5% and `gpt-5.6-terra` scored 98.1% semantic accuracy, against a 93.2% heuristic. The gaps are +2.3 and +4.9 points, short of the 8–10 point rule, so Gate 1 is a no. Macro accuracy (mean of per-domain scores) is reported beside the micro average so easy invoice-number columns do not decide the gate by themselves.

BENCH-C is a curated catalog of document fields with a human-specified gold regex for each column. It is not a scrape of private PDFs. Hidden negatives are near-miss mutations that fall outside that regex.
