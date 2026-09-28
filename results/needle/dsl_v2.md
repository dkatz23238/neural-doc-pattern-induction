## DSL v2

Character sets and integer ranges are read off the four supports and added only when assembly asks for them. Synthesis v1 stays the default, so the recorded exact-slice gates are unchanged. The recognizer is not retrained. BENCH-E stays sealed.

Of 554 BENCH-C patterns, 429 are exact, 100 are open, and 25 stay outside. Exact or open coverage is 0.955. The 25 outside patterns are word classes, space classes, month-name alternations, and five IPv4 addresses whose octet tree is still 20 nodes.

The 154 partial-range and mixed-class tasks are the measurement set. 149 of them are expressible. Assembly timeouts are 0. Empty charts on expressible tasks: 0. Max chart 43130 programs. Wall time 5943s.

Synthesis oracle on these 154 tasks is 0.571 (88/154). Deterministic facts alone reach 0.481. The neural chart supplies the rest.

A broad reading, the widest character set on each span with integer hulls removed, is hidden-set perfect on 0.383 of the 154. Every one of those charts contains that program (1.000 recoverable synthesis coverage). The broad reading is lower than the synthesis oracle because an integer hull or a neural program is sometimes the one that matches the hidden set. The remaining misses are unobserved numeric endpoints, supports cut at 32 characters, or a shape these facts do not name.

Selector accuracy on the 154 is 0.787. Given a hidden-set-perfect program, and excluding selector timeouts, accuracy is 0.927. Selector timeouts are 30. They are stability resynthesis on large charts. Assembly itself finished.

Headline semantic accuracy, with the frozen exact-slice scores kept and these 154 tasks rescored, is 0.885. Selector timeouts abstain at 0.5 inside that headline.

| Metric | Value |
| --- | --- |
| DSL exact/open coverage | 0.955 |
| Expressible partial/mixed tasks | 149/154 |
| Broad-set recoverable oracle | 0.383 |
| Recoverable synthesis coverage | 1.000 |
| Synthesis oracle on partial/mixed | 0.571 |
| Selector given perfect candidate | 0.927 |
| BENCH-C semantic accuracy | 0.885 |
| Empty charts on expressible tasks | 0 |
| Assembly timeouts | 0 |
| Selector timeouts | 30 |

