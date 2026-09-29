# Other flows

These commands are not required to call `needle()` or to rerun the compositional prior. They reproduce the frozen stack and the earlier measurements. Run them from the repo root with `uv run python`. `PYTHONPATH=. python` is the same entry point when the packages are importable.

BENCH-C is a diagnostic set. Training and checkpoint selection do not use it. BENCH-E stays sealed: `tests/test_bench_e.py` checks the seal, and `scripts/run_needle.py --stage summary` prints that it was not scored.

## Frozen checkpoints

`needle()` reads `results/scope/p2b/seed_0/checkpoint.pt` and `results/select/s1/seed_0/checkpoint.pt`.

```bash
uv run python scripts/run_scope.py --stage train
uv run python scripts/run_select.py --stage train
```

Those two sit on an earlier recognizer. Each of these writes a summary beside its checkpoint:

```bash
uv run python scripts/run_local.py --stage train
uv run python scripts/run_version_space.py
uv run python scripts/run_programs.py --stage train
```

| Script | Result |
| --- | --- |
| `scripts/run_local.py` | `results/local_chart/summary.md` |
| `scripts/run_version_space.py` | `results/version_space/summary.md` |
| `scripts/run_programs.py` | `results/programs/summary.md` |
| `scripts/run_scope.py` | `results/scope/summary.md` |
| `scripts/run_select.py` | `results/select/summary.md` |

`--stage summary` on the train scripts rewrites the markdown from the saved metrics and does not train again.

## Needle measurements

`scripts/run_needle.py` scores the frozen selector. Every stage ends by rewriting `results/needle/summary.md`.

```bash
uv run python scripts/run_needle.py --stage gate
uv run python scripts/run_needle.py --stage stress
uv run python scripts/run_needle.py --stage bench
uv run python scripts/run_needle.py --stage split
uv run python scripts/run_needle.py --stage transfer
uv run python scripts/run_needle.py --stage coverage
uv run python scripts/run_needle.py --stage summary
```

`--stage bench` scores BENCH-C once. The pages that `summary` concatenates also have their own entry points:

```bash
uv run python -m pattern_induction.latent_eval
uv run python -m pattern_induction.recall_eval
uv run python -m pattern_induction.dsl_v2_eval
uv run python -m pattern_induction.selector_modes
uv run python -m pattern_induction.coverage_eval
```

| Module | Page |
| --- | --- |
| `pattern_induction.latent_eval` | `results/needle/latent_oracle.md` |
| `pattern_induction.recall_eval` | `results/needle/recall_oracle.md` |
| `pattern_induction.dsl_v2_eval` | `results/needle/dsl_v2.md` |
| `pattern_induction.selector_modes` | `results/needle/selector_modes.md` |
| `pattern_induction.coverage_eval` | `results/needle/coverage.md` |
| `scripts/run_needle.py --stage transfer` | `results/needle/transfer.md` |
| `scripts/run_needle.py --stage split` | `results/needle/bench_split.md` |

## Prior trainer

`python -m prior.train` and `train_prior.py` are one loop. Size, device, and step count are configuration.

```bash
uv run python -m prior.train --config configs/prior_10k.yaml device=cpu name=prior_div_10k seed=1
uv run python train_prior.py --device cpu --model-size 10k --steps 20000 --batch-size 32
```

Overrides are `key=value` or the flags `--device`, `--model-size`, `--steps`, `--batch-size`, and `--mixed-precision`. `name=` chooses the checkpoint directory. `max_train=` keeps a prefix of the training shard. `model_size` is `0`, `10k`, `30k`, `100k`, `300k`, `1m`, `3m`, or `5m`. `model_size` selects the flat MLP. The AST and factor configs set `kind` themselves and must not be combined with that override. The menu-sampler curve, including 300k, is `results/prior/scaling.md`.

`python -m prior.generate --pilot 8` prints per-family keep rate without writing shards. The concept factory is the current experiment; its command is in the README. The expansion prior is `results/prior/expansion.md`.

```bash
uv run python -m prior.expansion --data results/prior/shards --seeds 0 1
```

The packet-identifiability audit is `results/prior/packets.md`.

```bash
uv run python -m prior.packet_audit --cache /tmp/relative-evidence --out results/prior/packets.json
```

The candidate-relative scorer is `results/prior/relative.md`.

```bash
uv run python -m prior.relative --data results/prior/shards --seeds 0 1
```

The constituent probes and the 2,863-parameter matcher are `results/prior/constituents.md`.

```bash
uv run python -m prior.constituent_audit --data results/prior/shards --out results/prior/constituents.json
uv run python -m prior.constituent_audit --relations --data results/prior/shards --out results/prior/constituents.json
uv run python -m prior.constituent_audit --slices --data results/prior/shards --out results/prior/constituents.json
uv run python -m prior.constituent_audit --numeric --data results/prior/shards --out results/prior/constituents.json
uv run python -m prior.constituent_audit --compact --data results/prior/shards --out results/prior/constituents.json
uv run python -m prior.constituent_match --data results/prior/shards --seeds 0 1
```

The pooled-vector audit is `results/prior/probes.md`.

```bash
uv run python -m prior.audit --data results/prior/shards --out results/prior/probes.json
```

The typed-relation scorer is `results/prior/relations.md`. The 15k model missed the compositional gates, so the 40k and 100k configs were not trained.

```bash
uv run python -m prior.train --config configs/prior_rel_15k.yaml device=cpu
```

The additive factor scorer is `results/prior/factor.md`.

```bash
uv run python -m prior.train --config configs/prior_factor_small.yaml device=cpu
uv run python -m prior.train --config configs/prior_factor_medium.yaml device=cpu
```

The flat 50-d candidate vector on this corpus is `results/prior/diversity.md`.

```bash
uv run python -m prior.train --config configs/prior_0.yaml device=cpu name=prior_div_0
uv run python -m prior.train --config configs/prior_10k.yaml device=cpu name=prior_div_10k
uv run python -m prior.train --config configs/prior_30k.yaml device=cpu name=prior_div_30k
uv run python -m prior.train --config configs/prior_100k.yaml device=cpu name=prior_div_100k
uv run python -m prior.train --config configs/prior_10k.yaml device=cpu name=prior_div_10k_n5000 max_train=5000
```

The recursive AST encoder on the same corpus is `results/prior/ast.md`.

```bash
uv run python -m prior.train --config configs/prior_ast_small.yaml device=cpu
uv run python -m prior.train --config configs/prior_ast_medium.yaml device=cpu
uv run python -m prior.train --config configs/prior_ast_large.yaml device=cpu
```

## How the frozen stack was reached

Each row is one study. The result page holds the numbers.

| Command | Page |
| --- | --- |
| `scripts/run_operators.py --stage train --arch h3o --recipe d0` | `results/operator_algebra/summary.md` |
| `scripts/run_hierarchy.py --stage train --arch h2 --recipe d0` | `results/hierarchy/summary.md` |
| `scripts/run_data_recipes.py --stage train --recipe d0` | `results/data_recipe/summary.md` |
| `scripts/run_compose.py` | `results/exp2/compose.md` |
| `scripts/run_renaming.py` | `results/exp2/renaming.md` |
| `scripts/run_membership.py` | `results/exp2/membership.md` |
| `scripts/run_schema.py` | `results/exp2/schema.md` |
| `scripts/run_override.py` | `results/exp2/override.md` |
| `scripts/run_selector.py` | `results/exp2/selector.md` |
| `scripts/run_structural_oracle.py` | `results/exp2/structural_oracle.md` |
| `scripts/run_ranker_pilot.py` | `results/exp2/ranker_pilot.md` |
| `scripts/run_candidate_oracle.py` | `results/exp2/candidate_oracle.md` |

The data-recipe teacher is optional and cached. The spend cap in code is $12.

```bash
uv run python scripts/run_data_recipes.py --stage teacher --prompt composition_v1.md --kind schemas
uv run python scripts/run_data_recipes.py --stage train --recipe d0
```

Prompts live in `data/teacher/prompts/`. Cached calls live under `teacher_runs/` and `data/teacher/`. Hierarchy and operator runs reuse that cache.

## Benchmarks and the first symbolic run

```bash
uv run python scripts/generate_benchmarks.py
uv run python scripts/run_experiment1.py
uv run python scripts/render_exp1_html.py
```

`generate_benchmarks.py` writes a review-sized suite (about 550 BENCH-C tasks, 1,500 BENCH-B, 320 BENCH-D, 1,200 BENCH-A) under `data/`. The full BENCH-A split is `--bench-a-dev 100000 --bench-a-val 10000 --bench-a-test 10000`. Shape buckets are hashed, so the larger draw keeps the same held-out shapes.

| Path | Role |
| --- | --- |
| `pattern_induction/evaluate.py` | Compile a regex and score hidden positives and negatives |
| `pattern_induction/heuristic.py` | Structural heuristic |
| `pattern_induction/baselines.py` | Memorization, grex, fAST |
| `pattern_induction/bench_c.py` | Curated document tasks |
| `pattern_induction/bench_b.py` | Document-style synthetic tasks |
| `pattern_induction/bench_a.py` | Controlled AST tasks with a shape holdout |
| `pattern_induction/bench_d.py` | The same schemas at 2, 3, 5, 10, or 20 examples |
| `results/exp1/exp1_report.html` | Gate 1 findings page |

Grex configs are `grex_default`, `grex_digits` (`-d`), `grex_words` (`-w`), `grex_repetitions` (`-r`), and `grex_combined` (`-d -w -r`). Install the binary with `cargo install grex` before that run.

fAST (`find-abstract-syntax-tree`) searches character by character. The runner records timeouts. Its concatenation operator is translated into a Python regex before scoring. Graphviz is optional.

```bash
uv run python scripts/run_experiment1.py --include-fast --fast-timeout 6 --fast-limit 20
```

## Frontier models

Gate 1 scores two frontier models on BENCH-C with the simple prompt.

```bash
uv run python scripts/run_frontier_bench_c.py
```

`pattern_induction/frontier.py` holds the prompts. No call is made without an API key. `results/exp1/frontier_bench_c.json` records the model id, date, and task count.
