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

The 1,158-parameter evidence scorer stays frozen. This run asks whether a separate prior, 1,765 parameters, can prefer an unsupported but plausible generalization when that scorer cannot separate the candidates. Its inputs are properties of the proposed expansion: unseen symbols and the superclass, width slack, integer slack and round or conventional endpoints, and a looser repeat. The program score is the evidence score plus λ times that prior. The prior is trained only on those ambiguous expansions.

The packed statistics are local and are not committed. A new checkout regenerates the shards, then the evidence cache, before the prior.

```bash
uv run python -m prior.generate --train 50000 --val 10000 --ood 20000 --seed 2507
uv run python -m prior.relative --data results/prior/shards --seeds 0 1
uv run python -m prior.expansion --data results/prior/shards --seeds 0 1
```

The prior is CPU-only. It does not train the 1.226M encoder and it does not change `needle()`. The packet audit and the evidence scorer are in [DOCS.md](DOCS.md).

The written record is already in the tree. Regenerating it only concatenates the result pages, including the sealed BENCH-E line:

```bash
uv run python scripts/run_needle.py --stage summary
```

| Page | What it records |
| --- | --- |
| `results/needle/summary.md` | Needle, then the prior curves |
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

The prior expands charsets and stretches closed widths. It does not prefer an open width, a loose repeat, or the probe-perfect integer window. Validation top-1 is 0.730 / 0.728 against the frozen scorer's 0.721 / 0.717. Holdout is 0.474 / 0.447 against 0.492 / 0.469. `prefix_charset_optional` stays 0. BENCH-E stays sealed.
