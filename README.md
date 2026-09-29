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

The question is whether the float vector given to a candidate node determines the gold label. The audit hashes that vector and reports collisions, conditional entropy, majority-label accuracy, and the Bayes top-1 ceiling of the candidate pool. The 1,158-parameter scorer is unchanged.

The packed statistics are local and are not committed. A new checkout regenerates the shards, then the scorer cache, before the audit.

```bash
uv run python -m prior.generate --train 50000 --val 10000 --ood 20000 --seed 2507
uv run python -m prior.relative --data results/prior/shards --seeds 0 1
uv run python -m prior.packet_audit --cache /tmp/relative-evidence --out results/prior/packets.json
```

The audit is CPU-only. It does not train the 1.226M encoder and it does not change `needle()`. The local scorer and the earlier probes are in [DOCS.md](DOCS.md).

The written record is already in the tree. Regenerating it only concatenates the result pages, including the sealed BENCH-E line:

```bash
uv run python scripts/run_needle.py --stage summary
```

| Page | What it records |
| --- | --- |
| `results/needle/summary.md` | Needle, then the prior curves |
| `results/prior/packets.md` | Identifiability of the local packets |
| `results/prior/relative.md` | Candidate-relative node scores |
| `results/prior/constituents.md` | Constituent probes and the candidate-conditioned matcher |
| `results/prior/probes.md` | Probes on the frozen 160-d support vector |
| `results/prior/relations.md` | Typed field, optional, and affix relations |
| `results/prior/factor.md` | Sum of local factor scores, CPU |
| `results/prior/ast.md` | Recursive candidate encoder |
| `results/prior/diversity.md` | Compositional sampler and the flat 10k / 30k / 100k sweep |
| `results/prior/scaling.md` | Earlier menu-sampler curve |

Tight repeat bounds have conditional entropy 0.030, and the scorer matches that ceiling. Charset expansion has gold recall 0.263, and `prefix_charset_optional` has a packet ceiling of 0.000. BENCH-E stays sealed.
