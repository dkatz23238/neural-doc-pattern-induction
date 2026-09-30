# Needle

Needle turns a few positive example strings into a compact regular expression. The score is semantic: hidden positives should match, and hidden negatives should not. Two examples are the minimum. One example abstains.

v1's decision is the code-length rule on eight candidates, the selector that scores 0.9625 on BENCH-C. The frozen recognizer reports confidence and alternatives. It does not replace that regex. The 2,145-parameter override in `pattern_induction/safe_override.py` is an experiment and is not loaded.

## Install

[uv](https://docs.astral.sh/uv/) is the environment. From a fresh checkout:

```bash
uv sync
```

That installs this repo in editable mode with torch, NumPy, PyYAML, pyarrow, and `find-abstract-syntax-tree==0.2.1`. `uv run` does not need `PYTHONPATH`.

`pip install -r requirements.txt` installs only the fAST pin. With the packages on `PYTHONPATH`, the same commands below work as `PYTHONPATH=. python ...`.

## Use it

Run from the repo root. The call reads:

- `results/scope/p2b/seed_0/checkpoint.pt`
- `results/select/s1/seed_0/checkpoint.pt`

```bash
uv run needle CUS-DE1234 CUS-SO9281 CUS-AM0042
uv run python -c 'from pattern_induction.needle import needle; print(needle(["CUS-DE1234", "CUS-SO9281", "CUS-AM0042"]))'
```

`examples/showcase.py` runs the six field examples and prints regex, confidence, and alternatives. Checkpoints are the files named above; the code-length rule does not load the override.

```python
from pattern_induction.needle import needle

result = needle(["CUS-DE1234", "CUS-SO9281", "CUS-AM0042"])
result.regex          # symbolic winner
result.confidence     # does not change result.regex
result.ambiguous
result.alternatives
```

The selector sees every example, whole. The recognizer reads the first four, each clipped to 32 characters. A missing checkpoint still returns the symbolic regex; confidence then uses only the code-length margin. Alternatives are other languages, not spellings of the winner. Retraining the recognizer is in [DOCS.md](DOCS.md). The release note is `results/needle/v1.md`. `scripts/profile_needle.py` times the six showcase calls and checks that their regexes are unchanged.

```bash
uv run python -m unittest discover -s tests
```

## Current experiment

Track B is a teacher-driven concept bank. A Terra call invents realistic string fields. The raw response is committed before parsing. A concept is kept only when Needle's executor accepts every positive and rejects every near miss. The large training shards are not stored; `data/factory/manifests/factory_v1.json` names the bank hash, the parser version, and the seed.

gpt-6-terra is not on this key. Calls use gpt-5.6-terra. The spend cap in this module is $10, separate from the older recipe-teacher cap.

```bash
uv run python -m pattern_induction.factory --stage generate --prompt real_world_concepts_v1.md --domain logistics
uv run python -m pattern_induction.factory --stage campaign --target 400
uv run python -m pattern_induction.factory --stage diversity
uv run python -m pattern_induction.factory --stage preview --limit 4
```

Prompts live in `data/factory/prompts/`. One call does not train `needle()`. The expansion prior and the packet audit are in [DOCS.md](DOCS.md).

The written record is already in the tree. Regenerating it only concatenates the result pages, including the sealed BENCH-E line:

```bash
uv run python scripts/run_needle.py --stage summary
```

| Page | What it records |
| --- | --- |
| `results/needle/summary.md` | Needle, then the prior curves |
| `results/needle/v1.md` | Frozen v1 decision, BENCH-C, and BENCH-E |
| `results/needle/showcase.md` | Six calls and what they return |
| `results/needle/inference.md` | Latency of the frozen decision after the inference pass |
| `results/factory/bank.md` | Accepted concept bank and its diversity |
| `results/factory/mix.md` | Frozen prior trained with the concept bank |
| `results/factory/typed.md` | One scale per ambiguity |
| `results/needle/safe_override.md` | Conservative override of the symbolic selector |
| `results/prior/expansion.md` | Tiny prior over ambiguous expansions |
| `results/prior/packets.md` | Identifiability of the local packets |
| `results/prior/relative.md` | Candidate-relative node scores |
| `results/prior/constituents.md` | Constituent probes and the candidate-conditioned matcher |
| `results/prior/probes.md` | Probes on the frozen 160-d support vector |
| `results/prior/relations.md` | Typed field, optional, and affix relations |
| `results/prior/factor.md` | Sum of local factor scores, CPU |
| `results/prior/ast.md` | Recursive candidate encoder |
| `results/prior/diversity.md` | Compositional sampler and the flat 10k / 30k / 100k sweep |
| `results/prior/scaling.md` | Earlier menu-sampler curve |

Sixty-six teacher calls, $3.07 of the $10 cap, left 418 executor-checked templates. Unique programs 325, AST shapes 118, semantic families 277, witnessed ambiguity 295, novel-composition share 0.502, top-1 program share 0.084. Whole concept families are assigned to train, validation, or held-out in `data/factory/manifests/split_v1.json`. Rejections are bucketed as bad teacher output, a DSL limitation, or a parser limitation.

The comparison against that bank is in `results/factory/mix.md`. The 1,158-parameter evidence scorer stays frozen. The 1,765-parameter prior is trained again with the same loss, with each batch drawn half from the synthetic ambiguous pool and half from the factory train split. Held-out template top-1 stays with the evidence scorer (mixture 0.417 / 0.438, evidence 0.472 / 0.458).

`results/factory/typed.md` gives each ambiguity its own scale. Factory held-out integer programs beat evidence, and open width no longer falls, but synthetic OOD integer programs still drop by more than 0.02. That is a negative result for a v1 prior: the teacher bank is not scaled further.

`results/needle/safe_override.md` keeps the 96.2% symbolic selector and tries a conservative override toward Needle's existing program. On BENCH-C the override scores 0.9633 against 0.9625, with 27 fixes and 26 regressions. The easy family dips from 0.9909 to 0.9862. That stops architecture work for v1.

`results/needle/v1.md` freezes the code-length rule as the decision. Recomputing it on BENCH-C matches the stored 0.9625. BENCH-E, opened once, scores 0.9848 on the same rule. The override stays in the repo and is not loaded.
