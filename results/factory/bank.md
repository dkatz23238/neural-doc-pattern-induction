## Concept factory

Terra proposes a field and a program. The raw response is stored before parsing. A concept enters the bank only when Needle's executor accepts every positive and rejects every near miss. Rejected concepts stay in `data/factory/rejected/`. Training shards are not committed; the manifest names the bank hash and the generator version that would rebuild them.

Parser `factory-2`. Teacher model `gpt-5.6-terra`. gpt-6-terra is not on this key. Templates 11. Unique latent programs 11. AST shapes 11. Concepts 11. Teacher domains 11. Semantic families 9. Bank hash `3607c03775cb5d65`.

Program entropy 3.459 bits. AST-signature entropy 3.459 bits. Semantic-family entropy 3.027 bits. Operator-combination entropy 2.914 bits. Domain entropy 3.459 bits. Top-1 program share 0.091. Top-10 operator-combination share 1.000. Top-10 semantic-family share 1.000. Mean positives per language 3.00. Witnessed ambiguity 10.

Novel-composition share waits on the compositional generator. This page counts teacher templates.

| Run | Prompt | Prompt domain | Accepted | Rejected |
| --- | --- | --- | ---: | ---: |
| 2026-09-29_terra_001 | real_world_concepts_v1.md | logistics | 6 | 0 |
| 2026-09-29_terra_002 | ambiguity_v1.md | logistics | 5 | 1 |

Calls 2. Spent $0.0919 of the $4 factory cap.

| Domain | Templates |
| --- | ---: |
| air_freight | 1 |
| commercial_invoice | 1 |
| container_shipping | 1 |
| delivery_appointment_references | 1 |
| expedited_load_references | 1 |
| freight_invoice_currency_fields | 1 |
| freight_rating | 1 |
| logistics | 1 |
| package_seal_identifiers | 1 |
| shipping_documentation | 1 |
| warehouse_slot_labels | 1 |

| Reject reason | Count |
| --- | ---: |
| unbound_literal | 1 |

BENCH-E stays sealed. `needle()` is unchanged.
