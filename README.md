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

The AST prior keeps the candidate generator, the compositional corpus, and the trainer fixed, and changes the candidate representation. A program is a tree. Optional, repeat, sequence, and alternation each have one shared composition function. Held out of training: optional prefix plus a character class, a repeated field plus a numeric interval, and a lexicon plus a literal suffix.

Shards are written under `results/prior/shards/` and are not committed. A new checkout regenerates them before training.

```bash
uv run python -m prior.generate --train 50000 --val 10000 --ood 20000 --seed 2507
uv run python -m prior.train --config configs/prior_ast_small.yaml device=cpu
uv run python -m prior.train --config configs/prior_ast_medium.yaml device=cpu
uv run python -m prior.train --config configs/prior_ast_large.yaml device=cpu
```

`seed=1` repeats a run. `device=cuda` is the same script when CUDA is present; a request for CUDA on a CPU machine exits. Checkpoints go to `results/prior/runs/<name>/seed_<n>/`. `train_prior.py` calls the same main. The flat compositional commands are in [DOCS.md](DOCS.md).

The written record is already in the tree. Regenerating it only concatenates the result pages, including the sealed BENCH-E line:

```bash
uv run python scripts/run_needle.py --stage summary
```

| Page | What it records |
| --- | --- |
| `results/needle/summary.md` | Needle, then the prior curves |
| `results/prior/ast.md` | Recursive candidate encoder, CPU |
| `results/prior/diversity.md` | Compositional sampler and the flat 10k / 30k / 100k sweep |
| `results/prior/scaling.md` | Earlier menu-sampler curve |

On the AST sweep, validation top-1 is 0.937 at 15,065 parameters. Held-out `repeat_interval` reaches 0.341 on the better seed, and `prefix_charset_optional` stays at 0.000. The shared encoder does not clear those compositions. BENCH-E stays sealed.
