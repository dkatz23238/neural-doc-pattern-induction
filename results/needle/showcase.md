# Showcase

Six calls to `needle()`. The regex is the code-length winner. Confidence and `ambiguous` come from the frozen recognizer and the cost gap. On every row below the recognizer proposed a different program, so the call is marked ambiguous and the symbolic regex stays.

| Call | Examples | Regex | Confidence | Seconds |
| --- | --- | --- | --- | --- |
| invoice | `INV-2024-10021`, `INV-2026-88310`, `INV-2025-00418` | `^INV-\d{4}-\d{5}$` | 0.25 | 0.67 |
| version | `1.2.0`, `1.4.3`, `2.0.1` | `^(?:\d\.){2}\d$` | 0.19 | 7.49 |
| section | `1.2`, `1.10`, `3.4` | `^\d\.\d{1,2}$` | 0.21 | 0.87 |
| filename | `report.pdf`, `notes.txt`, `budget.csv` | `^[a-z]{5,6}\.(?:pdf\|docx\|xlsx\|csv\|txt\|png\|jpg\|jpeg\|json\|xml\|html\|zip)$` | 0.25 | 2.05 |
| purchase order | `PO-1000`, `PO-1042`, `PO-1108` | `^PO-\d{4}$` | 0.24 | 0.18 |
| customer | `CUS-DE1234`, `CUS-SO9281`, `CUS-AM0042` | `^CUS-[A-Z]{2}\d{4}$` | 0.24 | 0.27 |

The selector itself takes 2–4 ms on these calls. The remaining time is the recognizer, which is loaded once and then reports the alternative. Resident memory after that load is about 355 MB.

Alternatives returned with the invoice call: `^INV(?:\-)?\d{4}\-\d{1,5}$`, `^INV-202[0-9]-\d{5}$`, `^INV-202\d-\d{5}$`, `^INV-202[0-9a-f]-\d{5}$`. None of them replaces the winner.
