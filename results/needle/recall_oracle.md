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

