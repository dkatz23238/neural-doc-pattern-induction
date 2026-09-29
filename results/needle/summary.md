# Needle CPU

The scope recognizer, the version-space chart, and the witness head stay frozen. A contingent delimiter, a punctuation mark that some supports omit, adds an optional suffix when it is the last separator. Shared separators do not fire. The witness head is not retrained.

Parameters: 1226128 (102741 in the witness head).

| Family | Semantic top-1 | Semantic accuracy | Version-space oracle | Languages |
| --- | --- | --- | --- | --- |
| primitive | 1.000 | 1.000 | 1.000 | 137.4 |
| repeat | 1.000 | 1.000 | 1.000 | 421.1 |
| optional | 1.000 | 1.000 | 1.000 | 70.4 |
| alternation | 1.000 | 1.000 | 1.000 | 75.0 |
| prefix_repeat | 1.000 | 1.000 | 1.000 | 718.3 |
| repeat_optional | 1.000 | 1.000 | 1.000 | 4.6 |
| prefix_alt | 1.000 | 1.000 | 1.000 | 91.9 |
| optional_internal | 1.000 | 1.000 | 1.000 | 613.5 |
| repeat_extra | 1.000 | 1.000 | 1.000 | 547.7 |

Atomic semantic top-1 1.000. Held-out semantic top-1 1.000. Held-out semantic accuracy 1.000. Held-out version-space oracle 1.000. Worst held-out family prefix_repeat 1.000. Renaming invariance 1.000.

semantic top-1 cleared the stretch line on held-out composition.

## Stress

Eight draws per generator setting. Stability stays on. Core cells are two supports, probes at repeat depth 6–7, a within-class rename onto a held-out mark, and width-2 supports whose language still allows a width-1 piece. A lower-case prefix and a prefix-plus-repeat-plus-optional tail are reported beside those cells.

| Cell | Role | Semantic top-1 | Oracle | Languages | Max languages |
| --- | --- | --- | --- | --- | --- |
| few_support | core | 1.000 | 1.000 | 1037.2 | 8708 |
| few_support / prefix_repeat | core | 1.000 | 1.000 | 3299.6 | 8708 |
| few_support / repeat_optional | core | 1.000 | 1.000 | 3.4 | 5 |
| few_support / prefix_alt | core | 1.000 | 1.000 | 76.1 | 96 |
| few_support / optional_internal | core | 1.000 | 1.000 | 221.2 | 222 |
| few_support / repeat_extra | core | 1.000 | 1.000 | 1585.8 | 2799 |
| long_probe | core | 1.000 | 1.000 | 474.6 | 983 |
| rename | core | 1.000 | 1.000 | 390.2 | 1190 |
| rename / prefix_repeat | core | 1.000 | 1.000 | 680.1 | 1190 |
| rename / repeat_optional | core | 1.000 | 1.000 | 5.4 | 16 |
| rename / prefix_alt | core | 1.000 | 1.000 | 85.8 | 109 |
| rename / optional_internal | core | 1.000 | 1.000 | 743.1 | 1151 |
| rename / repeat_extra | core | 1.000 | 1.000 | 436.5 | 745 |
| adversarial_width | core | 1.000 | 1.000 | 1213.2 | 2388 |
| lower_prefix | report | 1.000 | 1.000 | 463.7 | 2495 |
| lower_prefix / prefix_repeat | report | 1.000 | 1.000 | 845.5 | 2495 |
| lower_prefix / prefix_alt | report | 1.000 | 1.000 | 81.9 | 134 |
| nested | report | 0.500 | 0.500 | 2.0 | 6 |

Core semantic top-1 1.000. Core oracle 1.000. Worst core cell few_support 1.000. On the nested stack, top-1 equals the oracle. The misses are the draws whose optional hyphen is also the repeat separator: that character is in every support, so the contingent rule stays quiet and the frozen top-8 does not name the tail.

## BENCH-C

Scored once on 554 tasks, using the first four visible strings and at most 32 characters of each. Semantic accuracy 0.754. Macro 0.741. The old code-length rule on eight candidates scored 0.962. The structural pool oracle at 16 scored 0.995.

The split below separates grammar coverage, synthesis, and selection.

## BENCH-C split

The gold regex is compared with the program fragment: literals, digit / upper / lower, concatenation, optional, alternation, and repeat. A compound repeat stops at 8. A tree stops at 16 nodes. Partial ranges and mixed classes are outside the fragment. An unbounded quantifier is marked open, because the executor approximates it with that cap.

Of 554 tasks, 320 are exact (0.578), 60 are open (0.108), and 174 are outside the fragment (0.314). Selector accuracy on this pass is 0.754. The scoring run reported 0.754.

| Slice | Tasks | Selector accuracy | Selector perfect | Synthesis oracle | Selector perfect given synthesis | Mean languages | Mean seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exact | 320 | 0.788 | 0.222 | 0.353 | 0.628 | 272.0 | 6.06 |
| open | 60 | 0.770 | 0.100 | 0.400 | 0.250 | 548.4 | 9.21 |
| outside | 174 | 0.686 | 0.103 | 0.195 | 0.529 | 288.8 | 19.72 |
| fragment | 380 | 0.785 | 0.203 | 0.361 | 0.562 | 315.7 | 6.55 |

Outside the fragment: partial_range 89, mixed_class 65, nodes 10, space_class 5, word_class 5.

The version space is empty on 238 of 554 tasks. An empty chart abstains and scores 0.500. 118 of those empty tasks are exact in the fragment, out of 320 exact tasks. A program survives on the other 202 exact tasks. When synthesis puts a hidden-set-perfect program in the fragment, selector semantic accuracy is 0.971 and the chosen program is perfect on 0.562 of those tasks.

Against a perfect score, the exact slice contributes 12.2 points, tasks outside the fragment contribute 9.8, and unbounded quantifiers contribute 2.5. The exact-task gap is the empty charts. Selection, once a perfect program is present, stays at the semantic accuracy above. A few families also get large: filenames peak around 11 minutes, and dates and versions run to a few thousand languages.

| Domain | Tasks | Exact | Open | Outside | Selector accuracy | Synthesis oracle | Languages | Seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| account_reference | 25 | 25 | 0 | 0 | 0.859 | 0.240 | 250.2 | 7.78 |
| bank_reference | 15 | 10 | 0 | 5 | 0.667 | 0.067 | 25.1 | 2.16 |
| currency | 30 | 0 | 30 | 0 | 0.858 | 0.433 | 626.1 | 8.64 |
| customer_id | 30 | 30 | 0 | 0 | 0.566 | 0.133 | 25.5 | 0.28 |
| date | 25 | 0 | 0 | 25 | 0.849 | 0.120 | 988.1 | 26.94 |
| document_filename | 20 | 10 | 5 | 5 | 0.659 | 0.050 | 39.8 | 2.35 |
| doi | 5 | 0 | 0 | 5 | 0.750 | 0.400 | 30.2 | 1.32 |
| email | 15 | 0 | 5 | 10 | 0.692 | 0.267 | 87.0 | 6.16 |
| filename | 15 | 0 | 0 | 15 | 0.690 | 0.333 | 245.7 | 126.48 |
| form_field | 15 | 15 | 0 | 0 | 0.854 | 0.667 | 275.3 | 5.48 |
| hash | 20 | 0 | 0 | 20 | 0.500 | 0.000 | 0.0 | 0.06 |
| invoice_number | 24 | 20 | 0 | 4 | 0.959 | 0.333 | 66.9 | 5.77 |
| legal_numbering | 20 | 20 | 0 | 0 | 0.695 | 0.250 | 167.2 | 3.32 |
| log_line | 15 | 0 | 0 | 15 | 0.515 | 0.000 | 4.8 | 0.42 |
| measurement | 20 | 20 | 0 | 0 | 0.964 | 0.900 | 718.9 | 13.88 |
| ocr_field | 15 | 10 | 5 | 0 | 0.963 | 0.667 | 373.7 | 10.83 |
| order_id | 15 | 15 | 0 | 0 | 0.699 | 0.267 | 268.3 | 5.45 |
| page_reference | 15 | 15 | 0 | 0 | 0.975 | 1.000 | 622.1 | 12.72 |
| percentage | 10 | 10 | 0 | 0 | 0.990 | 0.900 | 1541.5 | 11.49 |
| phone | 20 | 20 | 0 | 0 | 0.844 | 0.300 | 117.3 | 2.93 |
| postal_code | 30 | 25 | 0 | 5 | 0.844 | 0.333 | 152.3 | 9.60 |
| scientific_id | 10 | 5 | 5 | 0 | 0.792 | 0.200 | 18.3 | 0.81 |
| section_id | 20 | 15 | 0 | 5 | 0.702 | 0.450 | 520.0 | 12.89 |
| serial | 15 | 0 | 0 | 15 | 0.657 | 0.200 | 28.9 | 1.42 |
| shipping_code | 5 | 5 | 0 | 0 | 0.500 | 0.000 | 0.0 | 0.04 |
| sku | 25 | 25 | 0 | 0 | 0.607 | 0.040 | 4.7 | 0.31 |
| spreadsheet_column | 15 | 10 | 0 | 5 | 0.897 | 0.333 | 227.9 | 20.61 |
| time | 20 | 5 | 0 | 15 | 0.923 | 0.600 | 1112.0 | 19.83 |
| tracking | 15 | 10 | 0 | 5 | 0.500 | 0.000 | 0.0 | 0.06 |
| transaction_id | 10 | 0 | 0 | 10 | 0.500 | 0.000 | 0.0 | 0.10 |
| url | 10 | 0 | 5 | 5 | 0.500 | 0.000 | 0.0 | 0.15 |
| uuid | 5 | 0 | 0 | 5 | 0.500 | 0.000 | 0.0 | 0.07 |
| version | 5 | 0 | 5 | 0 | 0.992 | 1.000 | 2354.4 | 47.66 |

## In-fragment transfer

The model and the grammar stay frozen. A nonempty chart is not rebuilt. An empty chart gains facts from an alignment of class runs: a shared text is a copy, a varying run is that class at the observed widths, a fixed run sequence that some supports omit is optional, and two unrelated skeletons are an alternation. Class width is not held to the repeat cap.

Exact tasks 320. Empty charts before 118, after 0 (0.000). Exact-slice synthesis oracle 0.697. Exact-slice selector accuracy 0.942. Headline selector accuracy, with previously empty tasks rescored and every other task left as scored, 0.843.

Of the 118 charts that were empty, 110 now contain a hidden-set-perfect program. The other 8 have a program and miss the hidden set because a width, a prefix, or a suffix never appeared in the four supports: five bank references, one phone width, one order-id prefix, and one section suffix.

Rescored 118 previously empty exact tasks in 2063s. Timeouts 12. Each of those 12 already had a hidden-set-perfect program; equivalence grouping did not finish, so the selector is counted as an abstention. On the 106 that finished, selector accuracy is 0.964.

| Alignment | Empty exact tasks | Filled | Hidden-set perfect | Selector accuracy |
| --- | --- | --- | --- | --- |
| equal | 101 | 101 | 94 | 0.918 |
| optional_prefix | 4 | 4 | 3 | 0.833 |
| optional_suffix | 9 | 9 | 9 | 0.914 |
| two_cluster | 4 | 4 | 4 | 0.979 |

## Visible coverage

The model and the grammar stay frozen. Alignment facts join a nonempty chart when they name a width, a literal, an optional region, or a binary alternation the frozen hypotheses do not already state. Equivalent concatenations collapse while the chart is built: adjacent copies, adjacent classes, a one-character class repeat, and an alternation branch that adds no strings. A chart the alignment does not extend is unchanged.

Exact tasks 320. Empty charts 0. Exact-slice synthesis oracle 0.922 (295/320). Previously perfect programs kept 113. Formerly empty charts that contain a hidden-set-perfect program 110. Of the 89 exact tasks that already had a program and no hidden-set-perfect one, 72 now contain one.

The other 25 exact misses, 17 from that 89 and 8 from the formerly empty charts, are positive-only identifiability failures. A bound endpoint, an alternative literal, or an optional text the gold uses does not appear in the four supports. None of the 25 is a visible structure the chart still fails to build. The recoverable-from-visible-support oracle is 0.922, the same number as the synthesis oracle.

| Missing fact | Tasks | What the four supports omit |
| --- | ---: | --- |
| unobserved bound | 16 | one endpoint of a class width, such as `{6,17}`, `{7,8}`, or `{1,3}` |
| unobserved branch | 8 | a literal alternative, such as a currency code, `N/A`, or a size |
| unobserved optional | 1 | an optional prefix that is present on every support |

Exact selector accuracy 0.966. Selector accuracy given a hidden-set-perfect program 0.974 on all 295 such tasks. Selector timeouts 0. The largest chart holds 36720 programs. Charts of that size are grouped by the regex they render, so distinct widths stay apart and the 90-second alarm does not fire. Headline selector accuracy, with these exact tasks rescored and every non-exact task left as scored, is 0.857.

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

## Selector witnesses

The recognizer stays the frozen 1.226M model. Selection no longer rebuilds the version space to score stability. A complete mixed character set is preferred to its observed hull by a probe that changes one character of a class the supports already show. An integer outside the observed endpoints abstains, so the digit class keeps that tie. BENCH-E stays sealed.

On the 154 partial-range and mixed-class tasks, selector timeouts are 0. Select latency is p50 0.780s, p95 4.175s, max 11.714s. Given a hidden-set-perfect chart, selector accuracy is 0.921. The BENCH-C headline, with these 154 scores replacing the previous ones, is 0.902.

Where mode A finished on 10 small charts, mode B agreed on 1.000 and mode C agreed on 1.000. The reported selector uses mode C plus the operator probes.

Old symbolic rule 0.962. Previous Needle 0.885. Oracle union before this pass 0.976. After this pass the Needle score is 0.902 and the oracle union is 0.977. Needle is strictly better on 79 tasks, the old rule on 291, and they tie on 184.

The router uses only inference-time signals: a witness decision or grammar confidence of at least 0.70, stability 1.0 on the direct check, and a chart smaller than 6000 that was not truncated. Otherwise it keeps the old rule. It takes Needle on 59 of the 154 tasks. Routed BENCH-C accuracy is 0.930. That is below the old rule, so this gate does not replace it.

Support-identifiable oracle, the observed integer hull and the observed character hull, is 0.091 perfect on the hidden set (mean accuracy 0.699). The hidden-set synthesis oracle is 0.571. The gap is the part a finite hidden sample can reward without the supports showing it.

| Metric | Value |
| --- | --- |
| Oracle union, old + previous Needle | 0.976 |
| Oracle union, old + this Needle | 0.977 |
| BENCH-C after selector timeouts | 0.902 |
| Selector given perfect candidate | 0.921 |
| Selector timeouts | 0 |
| Select p50 / p95 / max seconds | 0.780 / 4.175 / 11.714 |
| Routed accuracy | 0.930 |
| Support-identifiable oracle | 0.091 |
| Hidden-set synthesis oracle | 0.571 |

## Latent candidate oracle

The support-identifiable program is what the four examples justify. BENCH-C often asks for the latent rule those examples did not reveal. This pass adds generic rewrites — wider bounds, conventional numeric intervals, natural character superclasses, open repeats, alternations lifted to a class, and field structure read off the supports, including thousands groups, repeated fields, and optional affixes — and asks whether the hidden language is then in the set. No prior is trained. BENCH-E stays sealed.

Current synthesis oracle on 554 tasks is 0.796 (441/554). Expanded synthesis oracle is 0.915 (507/554). The rewrites recover 66 tasks the current chart missed. On the tasks that were reassembled, the mean chart has 1496 programs and the mean latent rewrite set has 286. Assembly timeouts: 0.

The support-identifiable program is hidden-perfect on 68 tasks. The other 486 fall into the ambiguity classes below. Those classes are the curriculum for a later prior, not extra benchmark answers.

| Ambiguity | Tasks |
| --- | ---: |
| unseen width/bound | 112 |
| unseen class member | 270 |
| unseen alternative | 42 |
| unseen optional branch | 1 |
| semantic numeric interval | 39 |
| open quantifier | 10 |
| other | 12 |

| Metric | Value |
| --- | --- |
| Current synthesis oracle | 0.796 |
| Expanded synthesis oracle | 0.915 |
| Recovered by expansion | 66 |
| Mean chart on reassembled tasks | 1496 |
| Mean latent rewrites on reassembled tasks | 286 |

The expanded oracle is not yet above 0.98, so these rewrites are not a basis for training a language prior. What remains is mostly closed lexicons, supports clipped at 32 characters, and optional branches that the four examples never show. BENCH-E stays sealed.

## Candidate recall

The legacy contract is unchanged: four examples, each clipped at 32 characters, the same chart as the latent-oracle pass. The product contract uses those four strings without that clip. Closed sets are proposed from a frozen bank (ISO 4217, Gregorian month names, RFC 1945 and the IANA status registry, common filename suffixes, mass units, letter sizes). An integer interval and a literal alternation now count as one node, so a 20-node IPv4 expression and a month-name alternation fit the 16-node cap. No prior is trained. BENCH-E stays sealed.

Legacy expanded oracle is 0.953 (528/554), up from 0.915 (21 gained, 0 lost). Product-contract oracle is 0.980 (543/554). Full strings add 15 tasks the clipped contract still misses. Expansion timeouts: 0.

Non-lexicon structural tasks: 0.979 (508/519). Closed-set candidate recall: 1.000 (35/35).

| Metric | Value | Gate |
| --- | ---: | ---: |
| Legacy BENCH-C expanded oracle | 0.953 | 0.95 |
| Full-input product-contract oracle | 0.980 | 0.98 |
| Non-lexicon structural oracle | 0.979 | 0.98 |
| Closed-set candidate recall | 1.000 | 0.95 |
| Expansion timeouts | 0 | 0 |

Tasks the product contract still misses, by domain:

- legal_numbering: 3
- section_id: 2
- bank_reference: 1
- document_filename: 1
- email: 1
- form_field: 1
- order_id: 1
- spreadsheet_column: 1

Three kinds of induction are in this stack. Visible structure is what the four strings literally support. Latent parametric generalization widens a width or a bound the sample only partly showed. Semantic set completion is the lexicon node: month names, currency codes, and the other frozen vocabularies, which a character class cannot invent.

The product-contract oracle is 0.980. Closed-set recall is 1.000. The structural oracle is 0.979 (508/519), short of its 0.98 gate. The 11 tasks still outside the set are optional branches and bounds the four examples do not show: a legal “bis”, an appendix number, a spaced ISO clause, a blank N/A, a third order prefix, a SWIFT branch code, a dotted local part, a matter filename stem, and a quarter digit whose superclass also accepts Q0 and Q8. A product-contract oracle above 0.98 is the bar for a learned prior. This pass does not train one. Those tasks stay out of the generator. BENCH-E stays sealed.

## Candidate prior

The candidate generator stays frozen at the 0.980 product-contract oracle. This pass trains only a scorer over candidates that generator already emitted. Each support set is embedded once by the frozen 1.226M encoder (160 dimensions). Each candidate is a vector of 50 structural features: operator counts, bounds, integer-range properties, lexicon identity, and coverage. The objective is listwise cross-entropy. Training loads parquet shards and does no regex synthesis. Device, precision, and parameter count are configuration of one script. This machine has no CUDA, so the curve below is the CPU run of that script.

The corpus is 50,000 training tasks, 10,000 validation tasks, and 20,000 structural holdout tasks, seed 2507. Generation kept 50,000 of 50,008 training draws (0.9998) and every validation and holdout draw. A task is stored only when the frozen expander emits the latent language, so the candidate-pool oracle on the shards is 1.000. Holdout families are unseen intervals, alphabets, lexicons, widths, and a prefix-plus-currency composition.

Checkpoints are chosen by validation top-1. BENCH-C is not a selection metric. BENCH-E stays sealed. `needle()` is unchanged.

| Parameters | Width | Hidden | Val top-1 | OOD top-1 | Score latency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 |  |  | 0.460 | 0.583 | 1.66 ms |
| 9,737 | 32 | 24 | 0.886 | 0.510 | 0.13 ms |
| 30,145 | 64 | 64 | 0.876 | 0.543 | 0.64 ms |
| 105,505 | 192 | 96 | 0.850 | 0.612 | 0.33 ms |
| 289,601 | 320 | 192 | 0.838 | 0.610 | 0.82 ms |

Val and OOD are the mean of seeds 0 and 1 at the best validation checkpoint. The 0-parameter row is a fixed symbolic score, identical across seeds. Latency is one CPU forward of 32 tasks with 32 candidates each.

Per seed, validation / holdout:

| Parameters | Seed 0 | Seed 1 |
| ---: | ---: | ---: |
| 0 | 0.460 / 0.583 | 0.460 / 0.583 |
| 9,737 | 0.883 / 0.498 | 0.889 / 0.522 |
| 30,145 | 0.875 / 0.548 | 0.877 / 0.538 |
| 105,505 | 0.839 / 0.649 | 0.862 / 0.575 |
| 289,601 | 0.835 / 0.630 | 0.841 / 0.589 |

The 9,737-parameter model is the knee. It learns the training families. Further parameters lower validation top-1. Holdout top-1 rises from 0.510 at 10k to 0.612 at 100k and stays there at 300k.

Where the 10k model moves, seed 0 / seed 1:

| Family | Symbolic | 10k validation | 10k holdout |
| --- | ---: | ---: | ---: |
| charset | 0.887 | 0.980 / 0.984 | 0.013 / 0.325 |
| interval | 0.215 | 0.715 / 0.713 | 0.037 / 0.082 |
| lexicon | 0.991 | 0.996 / 0.992 | 0.885 / 0.903 |
| prefix / composition | 0.206 | 0.753 / 0.762 | 1.000 / 0.972 |
| width | 0.004 | 0.973 / 0.994 | 0.557 / 0.330 |

The symbolic holdout column is charset 0.621, interval 0.391, lexicon 0.902, composition 1.000, width 0.000. The learned model picks up width, prefix, and in-distribution intervals. Held-out intervals stay near zero. Held-out alphabets and widths move with the seed.

Same 9,737-parameter model, 20,000 steps, smaller prefixes of the same 50,000 tasks:

| Training tasks | Val top-1 | OOD top-1 |
| ---: | ---: | ---: |
| 5,000 | 0.879 | 0.496 |
| 15,000 | 0.887 | 0.278 |
| 25,000 | 0.886 | 0.509 |
| 50,000 | 0.886 | 0.510 |

Validation accuracy is already at the 50,000-task level with 5,000 draws. The generator repeats a small menu of latent languages, so more copies of that menu do not add conventions.

Renaming invariance on 32 letter-prefix tasks is 1.000 for the candidate features, the frozen support embedding, and the MLP scores. Digit endpoints and lexicon tokens are semantic and were left in place.

| Gate | Measured |
| --- | ---: |
| Synthetic ambiguous top-1 > 0.90 | 0.889 |
| Synthetic candidate-pool oracle > 0.99 | 1.000 |
| OOD ambiguity > 0.85 | 0.612 |
| Renaming invariance | 1.000 |

CPU Gate A is open: in-distribution top-1 is 0.889 and holdout top-1 is 0.612. CPU Gate B: validation saturates at about 10k parameters, and this corpus saturates at about 5k tasks. The GPU gate is closed from this curve. A 1M–5M run is not scheduled. The recognizer stays at 1,226,128 parameters. BENCH-E stays sealed.

## Compositional prior

The candidate generator, the training script, and the model family stay frozen. This pass only changes how latent languages are sampled. A training language is a product of class, width, affix, optionality, repeat, interval, and lexicon factors. Three products are held out of training and validation: optional prefix plus a character class, a repeated field plus a numeric interval, and a lexicon plus a literal suffix. A row is stored only when the frozen expander emits that program. Checkpoints are chosen by validation top-1. The symbolic weights are the same fixed score as the previous curve. BENCH-C is not a selection metric. BENCH-E stays sealed. `needle()` is unchanged.

The corpus is 50,000 training tasks, 10,000 validation tasks, and 20,000 holdout tasks, seed 2507. Generation kept 50,000 of 50,008 training draws (0.9998), 10,000 of 10,001 validation draws, and 20,000 of 20,004 holdout draws. The candidate-pool oracle on the kept rows is 1.000.

| Split | Tasks | Unique latent ASTs | Entropy (bits) | Max AST share | Exact duplicate fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 50,000 | 1,881 | 9.544 | 0.0145 | 0.0217 |
| Validation | 10,000 | 1,444 | 9.407 | 0.0155 | 0.0075 |
| Holdout | 20,000 | 2,858 | 8.111 | 0.0302 | 0.0003 |

The most common training program is 1.45% of the corpus (`SEQ(lower{3,3},COPY(0:3-4),LEX[file_extension])`). The holdout families are absent from the training shards.

| Parameters | Width | Hidden | Val top-1 | OOD top-1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 |  |  | 0.315 | 0.390 |
| 9,737 | 32 | 24 | 0.888 | 0.296 |
| 30,145 | 64 | 64 | 0.888 | 0.279 |
| 105,505 | 192 | 96 | 0.864 | 0.424 |

Val and OOD are the mean of seeds 0 and 1 at the best validation checkpoint. The symbolic score does not use the seed, so both seed columns are that one evaluation. This machine has no CUDA.

Per seed, validation / holdout:

| Parameters | Seed 0 | Seed 1 |
| ---: | ---: | ---: |
| 0 | 0.315 / 0.390 | 0.315 / 0.390 |
| 9,737 | 0.886 / 0.292 | 0.891 / 0.299 |
| 30,145 | 0.889 / 0.297 | 0.888 / 0.261 |
| 105,505 | 0.845 / 0.540 | 0.883 / 0.308 |

Same 9,737-parameter model, 20,000 steps. The 50,000-task row is the mean above.

| Training tasks | Val top-1 | OOD top-1 |
| ---: | ---: | ---: |
| 5,000 | 0.875 | 0.226 |
| 50,000 | 0.888 | 0.296 |

Seed 0 / seed 1 at 5,000 tasks: validation 0.879 / 0.872, holdout 0.185 / 0.267.

Where the 10k model moves, at its best validation checkpoint, seed 0 / seed 1. The symbolic column is the 0-parameter score on this corpus.

| Family | Symbolic | 10k validation | 10k holdout |
| --- | ---: | ---: | ---: |
| charset_width | 0.644 | 0.983 / 0.989 |  |
| interval | 0.211 | 0.867 / 0.856 |  |
| lexicon | 0.779 | 0.947 / 0.945 |  |
| prefix_charset | 0.069 | 0.987 / 0.985 |  |
| prefix_interval | 0.112 | 0.867 / 0.867 |  |
| prefix_lexicon | 0.524 | 0.930 / 0.947 |  |
| prefix_optional | 0.000 | 1.000 / 1.000 |  |
| prefix_width | 0.170 | 0.932 / 0.946 |  |
| repeat_width | 0.240 | 0.360 / 0.392 |  |
| stem_lexicon | 0.635 | 0.996 / 0.994 |  |
| suffix_width | 0.154 | 0.926 / 0.940 |  |
| width | 0.550 | 0.876 / 0.884 |  |
| lexicon_suffix | 0.357 |  | 0.763 / 0.829 |
| prefix_charset_optional | 0.000 |  | 0.065 / 0.000 |
| repeat_interval | 0.814 |  | 0.049 / 0.068 |

`repeat_width` stays near 0.36 at 30k (0.355 / 0.357) and at 100k (0.364 / 0.369). On the holdout, `repeat_interval` is 0.090 / 0.133 at 30k and 0.353 / 0.171 at 100k, against the symbolic score of 0.814. `prefix_charset_optional` is 0.646 on 100k seed 0 and 0.000 on 100k seed 1.

Renaming invariance on 32 letter-prefix tasks is 1.000 for the candidate features, the frozen support embedding, and the MLP scores. Digit endpoints and lexicon tokens were left in place.

| Gate | Measured |
| --- | ---: |
| Validation top-1 > 0.90 | 0.888 |
| Structural OOD top-1 > 0.80 | 0.424 |
| Stretch OOD > 0.85 | 0.424 |
| Candidate oracle | 1.000 |
| Renaming invariance | 1.000 |
| 50k ahead of 5k | validation +0.013, holdout +0.070 |

The compositional corpus has 1,881 distinct training programs and 9.544 bits of signature entropy, and the learned prior still does not clear the holdout. The best mean holdout is 0.424 at 105,505 parameters, below 0.80, and the 9,737-parameter holdout of 0.296 is below the 0-parameter score of 0.390. Fifty thousand tasks raise validation from 0.875 to 0.888. The repeated-field families are the gap: in-distribution `repeat_width` stays near 0.36 at every size, and the held-out repeat-plus-interval program stays below the symbolic score. The next change is the candidate representation. The parameter sweep stops at 100k. BENCH-E stays sealed.

## AST candidate prior

The generator, the compositional corpus, the 160-d support vector, the loss, the optimizer, the 20,000 steps, and the validation checkpoint rule stay frozen. This pass changes the candidate representation. A program is a tree. A literal, a character class, an integer range, and a lexicon each have a learned leaf. Optional, repeat, sequence, and alternation each have one composition function, reused at every node of that operator. The scorer reads the 160-d support vector, a 32-d or 64-d tree vector, and eight relation features. Execution stays in the symbolic interpreter. Checkpoints are the best validation top-1. The symbolic weights are untouched. BENCH-C is not a selection metric. BENCH-E stays sealed. `needle()` is unchanged.

The corpus is the compositional one: seed 2507, 50,000 / 10,000 / 20,000 tasks, candidate oracle 1.000, 1,881 training programs, 9.544 bits. This machine has no CUDA. A run takes about five to six minutes.

| Parameters | Width | Hidden | Tree | Val top-1 | OOD top-1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 |  |  |  | 0.315 | 0.390 |
| 15,065 | 32 | 24 | 64 | 0.937 | 0.314 |
| 39,889 | 80 | 64 | 32 | 0.935 | 0.182 |
| 90,609 | 112 | 128 | 64 | 0.916 | 0.187 |

Val and OOD are the mean of seeds 0 and 1 at the best validation checkpoint. The 0-parameter row is the fixed symbolic score on this corpus. The flat compositional prior, for scale, was 0.888 / 0.296 at 9,737 parameters and 0.864 / 0.424 at 105,505.

Per seed, validation / holdout:

| Parameters | Seed 0 | Seed 1 |
| ---: | ---: | ---: |
| 0 | 0.315 / 0.390 | 0.315 / 0.390 |
| 15,065 | 0.937 / 0.300 | 0.938 / 0.328 |
| 39,889 | 0.935 / 0.178 | 0.935 / 0.186 |
| 90,609 | 0.910 / 0.146 | 0.922 / 0.227 |

The 39,889-parameter seed 0 checkpoint is step 18,000. Every other learned checkpoint is step 20,000.

Held-out families, seed 0 / seed 1, at that checkpoint. `repeat_width` is the in-distribution repeat family. The flat column is the 9,737-parameter compositional prior.

| Family | Symbolic | Flat 10k | AST 15k | AST 40k | AST 91k |
| --- | ---: | ---: | ---: | ---: | ---: |
| repeat_interval | 0.814 | 0.049 / 0.068 | 0.103 / 0.341 | 0.256 / 0.168 | 0.048 / 0.046 |
| prefix_charset_optional | 0.000 | 0.065 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| lexicon_suffix | 0.357 | 0.763 / 0.829 | 0.796 / 0.643 | 0.279 / 0.391 | 0.390 / 0.637 |
| repeat_width (validation) | 0.240 | 0.360 / 0.392 | 0.999 / 0.994 | 1.000 / 0.985 | 0.999 / 0.996 |

Every probe-perfect candidate set for `repeat_interval` contains both a repeat and an integer range (6,664 rows). Every probe-perfect set for `prefix_charset_optional` contains an optional and a character class (6,669 rows). On `lexicon_suffix`, 1,267 of 6,667 rows include a lexicon node among the probe-perfect candidates; the other gold programs are classes or sets that agree with the probe.

Renaming invariance on 32 letter-prefix tasks is 1.000 for the candidate features, the frozen support embedding, and the AST scores. Digit endpoints and lexicon tokens were left in place. The 15k seed 0 process recorded the score check before it included this encoder; reloading that checkpoint scores 1.000 on the same 32 tasks.

| Gate | Measured |
| --- | ---: |
| Validation top-1 >= 0.87 | 0.937 |
| Holdout top-1 > 0.60 | 0.314 |
| `repeat_interval` > 0.50 | 0.341 on the better seed |
| `prefix_charset_optional` > 0.50, both seeds | 0.000 / 0.000 |
| `lexicon_suffix` holds the flat 10k scores | 0.796 / 0.643 against 0.763 / 0.829 |
| Renaming invariance | 1.000 |

The shared repeat function learns the repeats that occur in training: validation `repeat_width` moves from about 0.36 on the flat prior to 0.99. The held-out combination does not follow it. `repeat_interval` rises on some seeds and stays below 0.50, then the 90,609-parameter model returns to the flat prior's range, against a symbolic score of 0.814. `prefix_charset_optional` is 0.000 at every size and both seeds. `lexicon_suffix` stays near the flat prior on one 15k seed and drops on the other seeds and the larger models. Mean holdout top-1 is 0.314 at 15,065 parameters, under the symbolic 0.390 and under 0.60, and larger trees score the holdout lower.

The generator still offers a perfect candidate on the kept rows, and the training corpus is the same 1,881-program distribution. Extra parameters in this encoder do not raise the holdout. The training-set size was left at 50,000. The tree was the only change, and it removes the in-distribution repeat failure while the three held-out operator combinations stay below the diagnostic gates. That does not support the claim that a shared recursive encoder is enough for those compositions. The 0.85 holdout target is untouched. BENCH-E stays sealed.

## Factorized candidate prior

The generator, the compositional corpus, the 160-d support vector, the optimizer, the 20,000 steps, and the validation checkpoint rule stay frozen. This pass changes the score. A candidate is a sum of local pieces: field, width, repeat, optional, affix, sequence, and alternation. Each piece has its own projection. The repeat projection does not receive the child, and the field projection does not receive the quantifier. An interaction term is `tanh(beta)` times the cosine of those two hidden states, with the states detached, so the term stays inside [-1, 1] and cannot rotate the pieces into a private code for one pair. Auxiliary targets mark whether the field, the width, the repeat, the optional, and the affix match some probe-perfect candidate. A rejected candidate that got one piece right still trains that piece. Execution stays symbolic. The symbolic weights are untouched. BENCH-C is not a selection metric. BENCH-E stays sealed. `needle()` is unchanged.

The corpus is the compositional one: seed 2507, 50,000 / 10,000 / 20,000 tasks. This machine has no CUDA. A run takes about a minute and a half.

| Parameters | Width | Val top-1 | OOD top-1 |
| ---: | ---: | ---: | ---: |
| 0 |  | 0.315 | 0.390 |
| 13,927 | 48 | 0.847 | 0.513 |
| 41,767 | 144 | 0.850 | 0.516 |

Val and OOD are the mean of seeds 0 and 1 at the best validation checkpoint. The 0-parameter row is the fixed symbolic score. The flat 9,737-parameter prior was 0.888 / 0.296. The AST prior was 0.937 / 0.314 at 15,065 parameters.

Per seed, validation / holdout:

| Parameters | Seed 0 | Seed 1 |
| ---: | ---: | ---: |
| 0 | 0.315 / 0.390 | 0.315 / 0.390 |
| 13,927 | 0.847 / 0.514 | 0.848 / 0.511 |
| 41,767 | 0.847 / 0.504 | 0.852 / 0.528 |

The 13,927-parameter checkpoints are step 16,000. The 41,767-parameter checkpoints are step 12,000 and step 16,000.

Held-out families, seed 0 / seed 1. `repeat_width` is the in-distribution repeat family. The AST column is the 15,065-parameter tree encoder. The flat column is the 9,737-parameter compositional prior.

| Family | Symbolic | Flat 10k | AST 15k | Factor 14k | Factor 42k |
| --- | ---: | ---: | ---: | ---: | ---: |
| repeat_interval | 0.814 | 0.049 / 0.068 | 0.103 / 0.341 | 0.730 / 0.701 | 0.692 / 0.766 |
| prefix_charset_optional | 0.000 | 0.065 / 0.000 | 0.000 / 0.000 | 0.000 / 0.003 | 0.000 / 0.005 |
| lexicon_suffix | 0.357 | 0.763 / 0.829 | 0.796 / 0.643 | 0.812 / 0.831 | 0.821 / 0.813 |
| repeat_width (validation) | 0.240 | 0.360 / 0.392 | 0.999 / 0.994 | 0.969 / 0.964 | 0.969 / 0.965 |

`charset_width`, which the AST prior scored at 0.985 / 0.974, is 0.542 / 0.539 at 14k and 0.541 / 0.539 at 42k. That drop is most of the validation gap. Lexicon, optional prefix, and stem families stay above 0.95.

Renaming invariance on 32 letter-prefix tasks is 1.000 for the candidate features, the frozen support embedding, and the factor scores. Auxiliary accuracy on the best 14k checkpoints is 0.920 validation and 0.712 holdout for seed 0, and 0.919 / 0.749 for seed 1.

| Gate | Measured |
| --- | ---: |
| Validation top-1 >= 0.87 | 0.847 |
| Holdout top-1 > 0.60 | 0.513 |
| `repeat_interval` > 0.50, both seeds | 0.730 / 0.701 |
| `prefix_charset_optional` > 0.50, both seeds | 0.000 / 0.003 |
| `lexicon_suffix` holds the flat 10k scores | 0.812 / 0.831 against 0.763 / 0.829 |
| Renaming invariance | 1.000 |

`repeat_interval` moves from 0.103 / 0.341 on the AST prior to 0.730 / 0.701 at 13,927 parameters, on both seeds, and stays in that range at 41,767. The symbolic score on the same rows is 0.814. `lexicon_suffix` holds the flat prior. `prefix_charset_optional` stays at zero. Mean holdout top-1 is 0.513, above the symbolic 0.390 and the AST 0.314, and below 0.60. The same width does not raise it. Validation is 0.847, below the AST 0.937, with `charset_width` near 0.54.

The additive repeat piece transfers to an integer range it was not trained on. The optional-prefix plus character-class combination does not. That split is the result of this objective. BENCH-E stays sealed.

## Typed relation prior

The generator, the compositional corpus, the 160-d support vector, the optimizer, the 20,000 steps, and the validation checkpoint rule stay frozen. The additive pieces stay too: field, width, repeat, optional, affix, sequence, and alternation. The repeat projection still does not see its child. This pass adds three relations and nothing else. Field pairs with width through a rank-8 product of those two hidden states. An optional pairs with the kind of constituent it governs. An affix pairs with the kind of field it attaches to, including whether it sits as a prefix, a suffix, or an infix. Scope embeddings start at zero, so an attachment absent from training adds nothing. Execution stays symbolic. The symbolic weights are untouched. BENCH-C is not a selection metric. BENCH-E stays sealed. `needle()` is unchanged.

The corpus is the compositional one: seed 2507, 50,000 / 10,000 / 20,000 tasks. This machine has no CUDA. The 40k and 100k widths were not trained. The 15k model missed the compositional gates, so there is no parameter curve and no data curve.

| Parameters | Width | Rank | Val top-1 | OOD top-1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 |  |  | 0.315 | 0.390 |
| 15,757 | 48 | 8 | 0.848 | 0.290 |

Val and OOD are the mean of seeds 0 and 1 at the best validation checkpoint, both step 16,000. The 0-parameter row is the fixed symbolic score. The additive prior at 13,927 parameters was 0.847 / 0.513.

Per seed, validation / holdout:

| Parameters | Seed 0 | Seed 1 |
| ---: | ---: | ---: |
| 0 | 0.315 / 0.390 | 0.315 / 0.390 |
| 15,757 | 0.847 / 0.317 | 0.848 / 0.264 |

The additive column is the 13,927-parameter factor prior.

| Family | Symbolic | Additive 14k | Relations 16k |
| --- | ---: | ---: | ---: |
| repeat_interval | 0.814 | 0.730 / 0.701 | 0.426 / 0.324 |
| prefix_charset_optional | 0.000 | 0.000 / 0.003 | 0.000 / 0.000 |
| lexicon_suffix | 0.357 | 0.812 / 0.831 | 0.525 / 0.467 |
| charset_width (validation) | 0.644 | 0.542 / 0.539 | 0.536 / 0.534 |
| repeat_width (validation) | 0.240 | 0.969 / 0.964 | 0.969 / 0.965 |

Renaming invariance on 32 letter-prefix tasks is 1.000 for the features, the frozen support embedding, and the relation scores.

| Gate | Measured |
| --- | ---: |
| Validation top-1 >= 0.87 | 0.848 |
| Holdout top-1 > 0.60 | 0.290 |
| `repeat_interval` >= 0.65, both seeds | 0.426 / 0.324 |
| `prefix_charset_optional` > 0.50, both seeds | 0.000 / 0.000 |
| `charset_width` > 0.80 | 0.536 / 0.534 |
| `lexicon_suffix` holds the additive scores | 0.525 / 0.467 against 0.812 / 0.831 |
| Renaming invariance | 1.000 |

In-distribution `repeat_width` stays at 0.969 / 0.965. Held-out `repeat_interval` falls from 0.730 / 0.701 to 0.426 / 0.324. `prefix_charset_optional` stays at zero. `charset_width` stays near 0.54. `lexicon_suffix` falls from 0.812 / 0.831 to 0.525 / 0.467. Mean holdout top-1 is 0.290, below the additive 0.513 and below the symbolic 0.390.

The three relations do not restore the missing compositions, and they cost the repeat transfer that the additive score had learned. The 40k model, the 100k model, and a larger corpus were not run. BENCH-E stays sealed.

## Support representation audit

The generator, the candidate pool, the frozen 160-d support vector, and the symbolic weights stay as they are. This pass does not train a new prior. It asks whether that vector still contains the facts a compositional prior would need. Each fact has its own probe, a linear map or one hidden layer of width 32, trained on the train families and chosen by validation balance. The held-out compositions are `repeat_interval`, `prefix_charset_optional`, and `lexicon_suffix`. Repeat, interval, optional, optional scope, affix, and lexicon are the family's latent facts. Charset and width are read from the strings. On this corpus every `prefix_charset_optional` support is an uppercase field whose string length changes with the prefix, and every `repeat_interval` support is a digit field whose string length changes with the repeat count. BENCH-C is not used. BENCH-E stays sealed. `needle()` is unchanged. The machine has no CUDA.

A string-side reader, used only to label charset and width and to drive the explicit match, agrees with the family bits: repeat 1.000, optional 0.9999, affix 0.9997 on the training shard, and repeat 1.000, optional 0.997, affix 0.997 on the holdout shard.

| Fact | Parameters | In distribution | Held-out composition |
| --- | ---: | --- | --- |
| Repeat | 322 | `repeat_width` 1.000 / 1.000 | `repeat_interval` 1.000 / 1.000 |
| Interval | 322 | `interval` 1.000 / 1.000, off on `width` 0.996 / 0.996 | `repeat_interval` 0.184 / 0.142 |
| Optional | 322 | `prefix_optional` 1.000 / 1.000, off on `prefix_charset` 1.000 / 1.000 | `prefix_charset_optional` 0.174 / 0.164 |
| Optional scope | 644 | `prefix_optional` 1.000 / 1.000 | `prefix_charset_optional` 0.104 / 0.119 |
| Charset | 644 | `charset_width` 0.994 / 0.992 | `prefix_charset_optional` 0.966 / 0.958 |
| Width | 483 | `repeat_width` 1.000 / 1.000, `prefix_optional` 1.000 / 1.000 | `repeat_interval` 1.000 / 1.000, `prefix_charset_optional` 0.187 / 0.179 |
| Affix | 483 | `suffix_width` 1.000 / 1.000, `prefix_charset` 1.000 / 1.000 | `lexicon_suffix` 0.000 / 0.001, `prefix_charset_optional` 0.475 / 0.693 |
| Lexicon | 322 | `lexicon` 0.948 / 0.980 | `lexicon_suffix` 0.307 / 0.313 |

The two numbers are seed 0 and seed 1 of the linear probe. Validation balance is 0.981 to 1.000 for every fact. The one-hidden-layer probe is about 5,200 parameters. It does not recover the missing bits. Optional scope on `prefix_charset_optional` is 0.000 / 0.000. Interval on `repeat_interval` is 0.000 / 0.269. The lexicon bit on `lexicon_suffix` rises to 0.851 / 0.806, and the suffix role stays at 0.000 / 0.000.

Read together, seed 0 of the linear probes gets both facts of a composition on this fraction of the holdout:

| Composition | Both facts | The pieces |
| --- | ---: | --- |
| `repeat_interval` | 0.184 | repeat 1.000, interval 0.184 |
| `prefix_charset_optional` | 0.104 | optional 0.174, scope 0.104, charset 0.966 |
| `lexicon_suffix` | 0.000 | lexicon 0.307, affix 0.000 |

Repeat is in the vector, including on a repeated integer field. The character class of the held-out optional family is in the vector. The interval bit is in the vector for a bare interval and a prefixed interval, and it is absent once that field is repeated. Optional scope is in the vector for an optional digit prefix, and it is absent once the field is an uppercase class. A suffix is in the vector after a digit field, and it is absent after a lexicon. Those are the compositions the prior could not learn.

The same strings, read before the pool, are then matched to each candidate. Repeat, optional scope, affix, field class, width, and lexicon each add 1 for an agreement and subtract 1 for a contradiction. A set that covers a pure observed class scores 0 on the field piece. The weights are fixed.

| Split | Top-1 | A gold shares the top score |
| --- | ---: | ---: |
| Validation | 0.288 | 0.694 |
| Holdout | 0.160 | 0.553 |
| `repeat_interval` | 0.195 | 0.982 |
| `prefix_charset_optional` | 0.065 | 0.319 |
| `lexicon_suffix` | 0.221 | 0.360 |

On every `prefix_optional` row the chosen candidate ties the gold on repeat, optional, affix, and field, and wins by 2 on width. The latent digit width is open. Every support field has one width, so an exact-width candidate outscores the gold. On `prefix_charset_optional` the winner ties the gold on optional, affix, and width, and wins by 0.681 on the field piece: the supports are uppercase, and a pure uppercase class outscores the alnum set. On `repeat_interval` a gold program shares the winning score on 0.982 of rows. The string facts treat a repeated digit class and a repeated integer range as the same numeric field, so argmax top-1 stays at 0.195.

The pooled vector is where optional scope and interval-under-repeat are lost. Exposing the string facts puts those bits back in front of a scorer, and the fixed match still misses probe-perfect top-1, because an integer window, a character the supports never show, and a width bound no support displays are not among those facts. The next change is the support representation: keep the local facts available, or retrain the encoder so they survive the pool. A larger prior on this vector would be scoring a representation that has already dropped those compositions. Pretrained semantic features become relevant once the structural facts decode and the remaining ambiguity is which closed set a token belongs to. This audit does not justify a GPU run.

## Constituent state

The recognizer stays the frozen 1.226M model. The pooled 160-d support vector is not the input. Each probe reads one constituent: the run layout the encoder already computed, or the numeric range of a digit run. Probes train on the train families and are chosen by validation balance. The held-out compositions are `repeat_interval`, `prefix_charset_optional`, and `lexicon_suffix`. The gate is accuracy above 0.90 on both seeds. BENCH-C is not used. BENCH-E stays sealed. `needle()` is unchanged. The machine has no CUDA.

A readout of every run at once is not a readout of one constituent. The joint layout still carries training co-occurrence. Every training interval is a single digit run, and every training suffix is uppercase.

| Readout | Interval on `repeat_interval` | Optional scope on `prefix_charset_optional` | Suffix on `lexicon_suffix` |
| --- | --- | --- | --- |
| Pooled 160-d vector | 0.184 / 0.142 | 0.104 / 0.119 | 0.000 / 0.001 |
| Run layout, all positions | 0.000 / 0.000 | 1.000 / 1.000 | 0.508 / 0.541 |
| Frozen span hidden | 0.052 / 0.007 | 1.000 / 1.000 | 0.000 / 0.000 |
| Class-blind layout | 0.052 / 0.016 | 1.000 / 1.000 | 0.738 / 0.745 |
| Last-run role | 0.000 / 0.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| Digit-run coverage | 1.000 / 1.000 | 1.000 / 1.000 | 0.000 / 0.000 |

The two numbers are seed 0 and seed 1 of a linear probe. Optional scope, the optional bit, repeat, and the character class are already at 1.000 / 1.000 on the unpooled run layout, including the held-out compositions. The pooled vector had optional scope at 0.104 / 0.119. The frozen span hidden keeps optional scope and drops the other two facts.

The suffix is the last run. Training suffixes are uppercase, so a probe of the whole layout uses that class and transfers at 0.508 / 0.541. Dropping class identity raises that to 0.738 / 0.745. Reading only the last run, still without class, transfers at 1.000 / 1.000, and the same probe scores the in-distribution suffix and the held-out optional prefix at 1.000 / 1.000. The role was in the layout. The other runs were the shortcut.

Interval is not in the layout and not in the span hidden. Within-field column constancy does not clear it either: a probe of those channels scores `repeat_interval` at 0.000 / 0.000. On the training shard an interval is one digit run, so the probe treats a second digit run as evidence against the label. The constancy values themselves are only a partial correlate. The fact that does transfer is the numeric coverage of a digit constituent, `(max − min + 1)` divided by the range of that width. One number, four parameters. `repeat_interval` is 1.000 / 1.000. In distribution, interval is 0.964 / 0.964, a repeated digit class is rejected at 1.000 / 1.000, and a spread digit width is rejected at 0.858 / 0.858. That last figure is the soft spot: some width tasks draw a tight handful of integers. The held-out repeated window is tighter than those, so the same threshold catches all of it.

The slices do not substitute for each other. Run presence scores optional scope at 1.000 / 1.000 and interval at 0.000 / 0.000. The repeat flag scores repeat at 1.000 / 1.000 and optional scope at 0.000 / 0.000. The field class scores the held-out character class at 1.000 / 1.000 and the suffix at 0.000 / 0.000. Digit coverage scores the suffix at 0.000 / 0.000.

The three held-out facts are available once each is stored on its constituent. They are not available as one vector, and interval and suffix are not in the frozen span hidden. A scorer that receives the pooled vector, or a concatenation it can use as a co-occurrence code, is still scoring a representation that has dropped the composition.

The matcher does not receive that vector. Each candidate factor queries one constituent: digit coverage for an integer, run presence for an optional, the last-run role for a literal, the repeat flag for a repeat, the field class for a class or a set, and a lexicon-column bit for a lexicon. The interaction is the product of the two projections, because a support term that is shared by every candidate cannot change the ranking. A residual reads the 50-d candidate features and does not see the support. The model has 2,863 parameters. There is one width, two seeds, 20,000 steps, batch 32, on CPU. The checkpoint is the best validation top-1.

| Gate | Seed 0 | Seed 1 |
| --- | ---: | ---: |
| `repeat_interval` | 0.425 | 0.392 |
| `prefix_charset_optional` | 0.000 | 0.000 |
| `lexicon_suffix` | 0.408 | 0.790 |
| `charset_width` | 0.979 | 0.982 |
| Holdout | 0.277 | 0.394 |
| Validation | 0.846 | 0.824 |
| Renaming invariance | 1.000 | 1.000 |

`charset_width` and renaming clear. Nothing else does. `repeat_interval` stays near 0.40 on both seeds, short of 0.70. `prefix_charset_optional` stays at 0. `lexicon_suffix` clears on one seed and not the other. Validation is 0.846 / 0.824, short of 0.87. In distribution, `repeat_width` is 0.284 / 0.260: the repeat query sees a count, not the repeat's bounds, so preservation of the repeat bit is not yet a ranking of repeat programs.

The constituent representation answers the probe question. Facts learned on separate constituents stay readable on the held-out compositions, above 0.90 on both seeds, when each fact is read from its own constituent. The 2,863-parameter matcher does not turn that into the selection gates. A larger scorer on the pooled vector is still the wrong next step. So is Terra, a pretrained backbone, or BENCH-E. The open question is how a candidate node should use a constituent it can already read.

## Candidate-relative scores

The constituent matcher can read that a run repeats. It cannot choose among `Repeat(2..4)`, `Repeat(1+)`, and `Repeat(exactly 2)`, because the repeat query sees a count and not the candidate's bounds. The same gap separates "this field is numeric" from `IntegerRange(2020, 2035)`.

This scorer gives each node the statistics of the span that node matches. A repeat node sees its own bounds and the observed firing counts. It does not receive the child's type. An integer node sees its own endpoints against the numbers it consumed. A charset node sees its alphabet against the characters it consumed. One network per node type, hidden width 16, 1,158 parameters. The program score is the sum. There is no support embedding and no cross-node term.

The local target is which hypothesis appears in a probe-perfect program. Local node selection is recorded first. Whole-program top-1 is the checkpoint criterion, best validation, two seeds, 20,000 steps, batch 32, CPU. `needle()` is unchanged. BENCH-E stays sealed.

### Local node selection

| Node | Family | What is selected | Seed 0 | Seed 1 |
| --- | --- | --- | ---: | ---: |
| Repeat | `repeat_width` | gold bounds | 0.990 | 0.990 |
| Repeat | `repeat_width` | observed min and max | 1.000 | 1.000 |
| Repeat | `repeat_interval` | gold bounds | 0.965 | 0.965 |
| Repeat | `repeat_interval` | observed min and max | 1.000 | 1.000 |
| Charset | `charset_width` | gold alphabet | 0.937 | 0.930 |
| Charset | `prefix_charset_optional` | gold alphabet | 0.000 | 0.000 |
| Integer | `interval` | gold window | 0.546 | 0.548 |
| Integer | `repeat_interval` | gold window | 0.167 | 0.137 |
| Integer | `repeat_interval` | observed hull | 0.000 | 0.000 |
| Lexicon | `lexicon_suffix` | gold lexicon | 1.000 | 1.000 |
| Optional | `prefix_optional` | gold optional | 1.000 | 1.000 |
| Literal | `repeat_interval` | gold literal | 1.000 | 1.000 |

On `repeat_interval` the chosen integer contains every observed value on 0.786 of seed 0, is one of the gold windows on 0.168, and is the observed hull on 0.000. A hull hypothesis is in the pool on every row. The gold windows are wider than that hull, about 4.7 of 15 hypotheses. The packet carries the slack, the expansion ratio, and a hull bit. The scorer uses containment. It does not pick the hull, and it does not pick the probe-perfect expansion.

The pooled validation repeat-node figure is 0.513 / 0.513. `prefix_charset`, `prefix_lexicon`, and `stem_lexicon` contribute gold open-repeats whose supports do not vary in count, and those rows score 0.000. The two repeat families themselves are the rows above.

### Whole-program top-1

| Gate | Seed 0 | Seed 1 |
| --- | ---: | ---: |
| `repeat_width` | 0.947 | 0.941 |
| `charset_width` | 0.961 | 0.957 |
| `repeat_interval` | 0.526 | 0.473 |
| `prefix_charset_optional` | 0.000 | 0.000 |
| `lexicon_suffix` | 0.952 | 0.933 |
| Validation | 0.721 | 0.717 |
| Holdout | 0.492 | 0.469 |
| Renaming invariance | 1.000 | 1.000 |

`repeat_width` is back above 0.90. The constituent matcher had it at 0.284 / 0.260. `charset_width` stays above 0.95. `repeat_interval` stays below the additive scorer's 0.730 / 0.701. The repeat half of that composition is already ranked at 1.000 tight bounds. The integer half is the miss, so the range packet does not yet carry the program.

`prefix_charset_optional` stays at 0. The supports are uppercase. Charset node selection on that family is 0.000. An alphanumeric class is an unseen digit, which this packet cannot observe.

`prefix_optional` programs are 0.087 / 0.065 while the optional node itself is 1.000 / 1.000. The miss sits on the width attached to the prefix, the same window-ranking gap as the integer head.

Validation is 0.721 / 0.717. In distribution, `interval` is 0.483 / 0.488 and `width` is 0.689 / 0.668. `lexicon_suffix` is 0.952 / 0.933, with the lexicon node at 1.000 / 1.000. Renaming invariance is 1.000 / 1.000.

Repeat bounds and the charset alphabet are now local scores of the span a node consumes. The remaining candidate-parameter gap is which containing integer window is the probe-perfect one. The unseen alphabet on `prefix_charset_optional` is a prior over characters the supports never show.

## BENCH-E

Sealed external split, 100 tasks, sha256 `f370a1707179ede5212d360751c5fb23ae7b6f507ec75665cd727dffc920ba9d`. Scored: no. Inspected: no. BENCH-C remains the diagnostic set.

