## Concept factory

Terra proposes a field and a program. The raw response is stored before parsing. A concept enters the bank only when Needle's executor accepts every positive and rejects every near miss. Rejected concepts stay in `data/factory/rejected/`. Training shards are not committed; the manifest names the bank hash and the generator version that would rebuild them.

Parser `factory-3`. Teacher model `gpt-5.6-terra`. gpt-6-terra is not on this key. Templates 29. Unique latent programs 28. AST shapes 24. Concepts 29. Teacher domains 15. Semantic families 27. Bank hash `2bde522c025cfb36`.

Program entropy 4.789 bits. AST-signature entropy 4.487 bits. Semantic-family entropy 4.694 bits. Operator-combination entropy 3.771 bits. Domain entropy 3.508 bits. Top-1 program share 0.069. Top-10 operator-combination share 0.793. Top-10 semantic-family share 0.414. Mean positives per language 3.00. Witnessed ambiguity 10.

Novel-composition share waits on the compositional generator. This page counts teacher templates.

| Run | Prompt | Prompt domain | Accepted | Rejected |
| --- | --- | --- | ---: | ---: |
| 2026-09-29_terra_001 | real_world_concepts_v1.md | logistics | 6 | 0 |
| 2026-09-29_terra_002 | ambiguity_v1.md | logistics | 5 | 1 |
| 2026-09-29_terra_003 | adversarial_v1.md | logistics | 6 | 0 |
| 2026-09-29_terra_004 | domain_expansion_v1.md | software | 2 | 4 |
| 2026-09-29_terra_005 | domain_expansion_v1.md | legal | 3 | 3 |
| 2026-09-29_terra_006 | domain_expansion_v1.md | finance | 4 | 2 |
| 2026-09-29_terra_007 | domain_expansion_v1.md | healthcare | 3 | 3 |

Calls 7. Spent $0.3198 of the $4 factory cap.

| Domain | Templates |
| --- | ---: |
| air_freight | 1 |
| commercial_invoice | 1 |
| container_shipping | 1 |
| delivery_appointment_references | 1 |
| expedited_load_references | 1 |
| finance | 4 |
| freight_invoice_currency_fields | 1 |
| freight_rating | 1 |
| healthcare | 3 |
| legal | 3 |
| logistics | 7 |
| package_seal_identifiers | 1 |
| shipping_documentation | 1 |
| software | 2 |
| warehouse_slot_labels | 1 |

| Reject reason | Count |
| --- | ---: |
| bad_width | 3 |
| positive_rejected | 2 |
| duplicate_example | 2 |
| near_miss_accepted | 2 |
| unbound_literal | 1 |
| bad_literal | 1 |
| bad_examples | 1 |
| bad_bounds | 1 |

BENCH-E stays sealed. `needle()` is unchanged.
