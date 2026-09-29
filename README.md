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

The typed-relation prior keeps the additive factor scores and adds three relations: field with width, optional with the constituent it governs, and affix with the field it attaches to. The repeat piece still does not see its child. Held out of training: optional prefix plus a character class, a repeated field plus a numeric interval, and a lexicon plus a literal suffix.

Shards are written under `results/prior/shards/` and are not committed. A new checkout regenerates them before training.

```bash
uv run python -m prior.generate --train 50000 --val 10000 --ood 20000 --seed 2507
uv run python -m prior.train --config configs/prior_rel_15k.yaml device=cpu
```

`seed=1` repeats a run. `device=cuda` is the same script when CUDA is present; a request for CUDA on a CPU machine exits. Checkpoints go to `results/prior/runs/<name>/seed_<n>/`. `train_prior.py` calls the same main. The additive, AST, and flat commands are in [DOCS.md](DOCS.md). `configs/prior_rel_40k.yaml` and `configs/prior_rel_100k.yaml` are the same scorer at larger widths. They were not trained: the 15k model missed the compositional gates.

The written record is already in the tree. Regenerating it only concatenates the result pages, including the sealed BENCH-E line:

```bash
uv run python scripts/run_needle.py --stage summary
```

| Page | What it records |
| --- | --- |
| `results/needle/summary.md` | Needle, then the prior curves |
| `results/prior/relations.md` | Typed field, optional, and affix relations |
| `results/prior/factor.md` | Sum of local factor scores, CPU |
| `results/prior/ast.md` | Recursive candidate encoder |
| `results/prior/diversity.md` | Compositional sampler and the flat 10k / 30k / 100k sweep |
| `results/prior/scaling.md` | Earlier menu-sampler curve |

On the 15,757-parameter relation model, validation top-1 is 0.848 and holdout top-1 is 0.290. Held-out `repeat_interval` is 0.426 / 0.324, below the additive prior's 0.730 / 0.701. `prefix_charset_optional` stays at 0.000 and `charset_width` stays near 0.54. BENCH-E stays sealed.
