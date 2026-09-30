# Showcase

Six calls to `needle()`. The regex is the code-length winner. Confidence mixes the code-length margin, whether the recognizer names the same language, and the one-bit ambiguity band. It does not replace the regex. Alternatives are other languages, not spellings of the winner.

| Call | Examples | Regex | Confidence | Ambiguous | Seconds |
| --- | --- | --- | --- | --- | --- |
| invoice | `INV-2024-10021`, `INV-2026-88310`, `INV-2025-00418` | `^INV-\d{4}-\d{5}$` | 0.70 | yes | 0.25 |
| version | `1.2.0`, `1.4.3`, `2.0.1` | `^(?:\d\.){2}\d$` | 0.52 | yes | 1.69 |
| section | `1.2`, `1.10`, `3.4` | `^\d\.\d{1,2}$` | 0.41 | yes | 0.33 |
| filename | `report.pdf`, `notes.txt`, `budget.csv` | `^[a-z]{5,6}\.(?:pdf\|docx\|xlsx\|csv\|txt\|png\|jpg\|jpeg\|json\|xml\|html\|zip)$` | 0.46 | yes | 0.49 |
| purchase order | `PO-1000`, `PO-1042`, `PO-1108` | `^PO-\d{4}$` | 0.82 | no | 0.13 |
| customer | `CUS-DE1234`, `CUS-SO9281`, `CUS-AM0042` | `^CUS-[A-Z]{2}\d{4}$` | 0.63 | yes | 0.16 |

The selector itself takes 2–4 ms. The remaining time is the recognizer. On the purchase order the recognizer's regex is a spelling of `^PO-\d{4}$`, so the call is not marked ambiguous and the program is attached. On the other five the recognizer names a different language, the symbolic regex stays, and confidence stays below certainty. Resident memory after the load is about 355 MB. Confidence is the fitted score from `results/needle/confidence.md`.

Invoice alternatives: `^INV(?:\-)?\d{4}\-\d{1,5}$`, `^INV-202[0-9]-\d{5}$`, `^INV-202[0-9a-f]-\d{5}$`, `^INV-+202\d-\d{5}$`.

Version alternatives: `^\d{1,2}(?:\.\d{1,2}){1,}$`, `^[0-9a-f]\.\d\.\d$`, `^\d?\.\d\.\d$`, `^\d\.\d\.\d{1,2}$`. The written-out form `^[0-9]\.\d\.\d$` is the same language as the winner, so it is not listed.

Purchase-order alternatives: `^PO-?\d{4}$`, `^PO-1\d{3}$`, `^PO-+\d{4}$`, `^PO-\d{2,6}$`.
