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
| `results/exp1/` | First symbolic run |

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

Grex configs are `grex_default`, `grex_digits` (`-d`), `grex_words` (`-w`), `grex_repetitions` (`-r`), and `grex_combined` (`-d -w -r`).

fAST is the ICGI 2023 reference implementation (`find-abstract-syntax-tree`). It searches character by character and often returns nothing within a few seconds on document-length strings. The runner records timeouts instead of inventing a regex. Its concatenation operator is translated into a Python regex before scoring.

## Frontier models

Gate 1 needs two frontier models on BENCH-C. This environment has no `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`, so those cells are empty on purpose. `results/exp1/frontier_prompts_sample.json` stores the simple prompt, the guarded prompt, and the structural-feature prompt from the spec. Temperature should be 0. Record the model id, date, and sample count with the scores.

## What this run does not do

No neural model is trained. The spec says to train only after Gate 1 shows an 8–10 point gap between the best symbolic system and frontier models on BENCH-C. Macro accuracy (mean of per-domain scores) is reported beside the micro average so easy invoice-number columns do not decide the gate by themselves.

BENCH-C is a curated catalog of document fields with a human-specified gold regex for each column. It is not a scrape of private PDFs. Hidden negatives are near-miss mutations that fall outside that regex.
