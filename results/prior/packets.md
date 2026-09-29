## Packet identifiability

The 1,158-parameter scorer ranks each node from one float vector: the span it consumed and the candidate's own parameters. This audit hashes that vector. Where the same vector recurs, it measures whether the gold label is a function of the vector: the collision rate, the conditional entropy in bits, the majority-label accuracy, and how often a gold row sits on a packet whose majority label is gold. The candidate-pool ceiling ranks each pool by the training conditional and breaks ties uniformly. Whole-program ceilings use the multiset of node packets in the same way.

The abstain vector is one constant row, so it is left out of the packet table. A packet that occurs once is identifiable only by memorizing that row. The recurring rows are the evidence that actually repeats.

### Where the packet determines the label

| Evidence | Entropy | Majority | Gold recall | Pool ceiling | Scorer |
| --- | ---: | ---: | ---: | ---: | ---: |
| Repeat, tight bounds | 0.030 | 0.996 | 1.000 | 0.990 / 0.965 | 0.990 / 0.965 |
| Lexicon name | 0.167 | 0.975 | 1.000 | 1.000 | 1.000 |
| Literal | 0.435 | 0.883 | 1.000 | 0.980–1.000 | 1.000 |
| Charset, exact alphabet | 0.183 | 0.949 | 0.711 | 0.829 | 0.937 / 0.930 |
| Optional prefix | — | — | — | 1.000 | 1.000 |

Repeat tight bounds are 13 packets. A few of those packets are occasionally not gold, which is the 0.300 collision rate, and the majority label is still gold on 0.996 of the rows. The pool ceiling on `repeat_width` is 0.996 in train and 0.990 on validation. On `repeat_interval` it is 0.965. The scorer lands on those same figures. Lexicon pools are 1.000 on `lexicon`, `stem_lexicon`, and `lexicon_suffix`. The optional node on `prefix_optional` is 1.000.

Whole-program ceilings from the multiset of those packets:

| Program | Packet ceiling | Scorer |
| --- | ---: | ---: |
| `repeat_width` | 0.942 | 0.947 / 0.941 |
| `charset_width` | 0.964 | 0.961 / 0.957 |
| `lexicon_suffix` | 0.963 | 0.952 / 0.933 |
| `width` | 0.676 | 0.689 / 0.668 |

The scorer is at the ceiling on the decisions the packets determine.

### Where the same packet is gold and a decoy

| Evidence | Entropy | Majority | Gold recall | Pool ceiling | Scorer |
| --- | ---: | ---: | ---: | ---: | ---: |
| Charset expansion | 0.334 | 0.891 | 0.263 | 0.000 on `prefix_charset_optional` | 0.000 |
| Charset, open width | 0.326 | 0.906 | 0.147 | — | — |
| Repeat, loose container | 0.766 | 0.642 | 0.017 | 0.000 | 0.000 |
| Integer hull, slack only | 0.419 | 0.910 | 0.000 | — | 0.000 hull picks |
| Integer expansion, recurring | 0.398 | 0.843 | 0.550 | 0.168 on val `interval` | 0.546 local |
| `prefix_optional` program | — | — | — | 0.168 | 0.087 / 0.065 |
| `repeat_interval` program | 0.000 inside OOD | 1.000 | — | 0.151 from the train table | 0.526 / 0.473 |
| `prefix_charset_optional` program | — | — | — | 0.031 | 0.000 |

Charset expansion is 2,010 recurring packets. The majority label is usually the decoy: a gold expansion row falls in a majority-gold packet only 0.263 of the time. `prefix_charset_optional` therefore has a charset-pool ceiling of 0.000, and the scorer scores 0.000. An open width has gold recall 0.147. A loose repeat, one that contains the observed counts without matching both bounds, has gold recall 0.017. The families whose gold mask includes that open repeat have a pool ceiling of 0.000.

Integer packets mostly do not recur. Of 207,781 integer rows, 132,483 are distinct, because the vector contains the absolute endpoints. Restricting to packets that occur at least twice, expansion still has entropy 0.398 and gold recall 0.550. Dropping the absolute endpoints and keeping slack, expansion ratio, containment, and the hull bit collapses the hull to 2 packets, and the majority label of both is the decoy (gold recall 0.000). The observed hull is not the gold label of its own packet. On validation `interval`, the train conditional ranks the pool at 0.168, or 0.265 on the rows whose gold packet was seen in training. On `repeat_interval` that seen-packet ceiling is 0.046. The scorer's local accuracy there is 0.167 / 0.137, and 0.546 / 0.548 on in-distribution `interval`. It is already above the identity table, by smoothing across endpoints the table has not memorized.

Inside the held-out split, a program signature that recurs has entropy 0.000 on `repeat_interval` and on `prefix_charset_optional`. The generator repeats a signature with one label. `prefix_charset_optional` has no program whose signature appeared in training, so the training conditional scores that split at 0.031. `repeat_interval` has 316 tasks whose signature was seen, with a ceiling of 1.000 on those tasks and 0.151 on the full split. The scorer reaches 0.526 / 0.473 by generalizing past the table. The additive scorer's 0.730 / 0.701 is higher still, on features that are not this packet. Nothing in the collision table says the local matcher is leaving a determined repeat or charset decision on the table.

`prefix_optional` shows the same split inside one family. The optional node ceiling is 1.000 and the scorer is there. The program ceiling is 0.168, and the scorer is at 0.087 / 0.065. The miss is the width attached to a prefix the packet does not pin down. `prefix_charset_optional` has an optional-node ceiling of 0.626 against a scorer at 0.253, and a program ceiling of 0.031, because the alphabet expansion is the unidentified piece.

The evidence layer is the 1,158-parameter scorer. It matches the packet ceiling on tight repeats, lexicons, literals, exact charsets, and the programs those decisions compose. Charset expansion, open width, and integer windows wider than the hull are not functions of the packet. Those are the decisions a separate prior would have to make. The local matcher is left as it is.
