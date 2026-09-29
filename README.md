# Needle

Needle turns a few positive example strings into a compact regular expression. The score is semantic: hidden positives should match, and hidden negatives should not. Two examples are the minimum. One example abstains.

`needle()` loads the frozen scope recognizer and the frozen witness head. It does not train. The learned candidate prior in `prior/` is a separate experiment; `needle()` still uses the grammar prior saved with the witness head.

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
uv run python -c 'from pattern_induction.needle import needle; print(needle(["CUS-DE1234", "CUS-SO9281", "CUS-AM0042"]))'
```

```python
from pattern_induction.needle import needle

result = needle(["CUS-DE1234", "CUS-SO9281", "CUS-AM0042"])
result.regex
result.alternatives
result.confidence
```

Supports are clipped to four strings and 32 characters. Retraining those two checkpoints is in [DOCS.md](DOCS.md).

```bash
uv run python -m unittest discover -s tests
```

## Current experiment

The question is whether a fact learned on one constituent is still there when that constituent shows up inside a composition the trainer did not see. The recognizer stays frozen. The pooled 160-d vector is not the input. Probes read the run layout and the numeric range of a digit run. Held out of training: optional prefix plus a character class, a repeated field plus a numeric interval, and a lexicon plus a literal suffix.

Shards are written under `results/prior/shards/` and are not committed. A new checkout regenerates them before the probes. The constituent cache is local and is not committed.

```bash
uv run python -m prior.generate --train 50000 --val 10000 --ood 20000 --seed 2507
uv run python -m prior.constituent_audit --data results/prior/shards --out results/prior/constituents.json
uv run python -m prior.constituent_audit --relations --data results/prior/shards --out results/prior/constituents.json
uv run python -m prior.constituent_audit --slices --data results/prior/shards --out results/prior/constituents.json
uv run python -m prior.constituent_audit --numeric --data results/prior/shards --out results/prior/constituents.json
uv run python -m prior.constituent_audit --compact --data results/prior/shards --out results/prior/constituents.json
uv run python -m prior.constituent_match --data results/prior/shards --seeds 0 1
```

The probes and the 2,863-parameter matcher are CPU-only. They do not train the 1.226M encoder and they do not change `needle()`. The pooled-vector audit and the earlier scorers are in [DOCS.md](DOCS.md).

The written record is already in the tree. Regenerating it only concatenates the result pages, including the sealed BENCH-E line:

```bash
uv run python scripts/run_needle.py --stage summary
```

| Page | What it records |
| --- | --- |
| `results/needle/summary.md` | Needle, then the prior curves |
| `results/prior/constituents.md` | Constituent probes and the candidate-conditioned matcher |
| `results/prior/probes.md` | Probes on the frozen 160-d support vector |
| `results/prior/relations.md` | Typed field, optional, and affix relations |
| `results/prior/factor.md` | Sum of local factor scores, CPU |
| `results/prior/ast.md` | Recursive candidate encoder |
| `results/prior/diversity.md` | Compositional sampler and the flat 10k / 30k / 100k sweep |
| `results/prior/scaling.md` | Earlier menu-sampler curve |

Optional scope on `prefix_charset_optional` is 1.000 / 1.000 from the unpooled runs. The suffix on `lexicon_suffix` is 1.000 / 1.000 from the last run. Interval on `repeat_interval` is 1.000 / 1.000 from the digit run's numeric coverage, and 0.000 / 0.000 from the run layout. BENCH-E stays sealed.
