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

## BENCH-E

Sealed external split, 100 tasks, sha256 `f370a1707179ede5212d360751c5fb23ae7b6f507ec75665cd727dffc920ba9d`. Scored: no. Inspected: no. BENCH-C remains the diagnostic set.

