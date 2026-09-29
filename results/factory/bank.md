## Concept factory

Terra proposes a field and a program. The raw response is stored before parsing. A concept enters the bank only when Needle's executor accepts every positive and rejects every near miss. Rejected concepts stay in `data/factory/rejected/`. Training shards are not committed; the manifest names the bank hash and the generator version that would rebuild them.

Parser `factory-4`. Teacher model `gpt-5.6-terra`. gpt-6-terra is not on this key. Templates 418. Unique latent programs 325. AST shapes 118. Concepts 412. Teacher domains 26. Semantic families 277. Bank hash `5cb161bb6f0da187`.

Program entropy 7.849 bits. AST-signature entropy 5.433 bits. Semantic-family entropy 7.395 bits. Operator-combination entropy 4.523 bits. Domain entropy 4.145 bits. Top-1 program share 0.084. Top-10 operator-combination share 0.706. Top-10 semantic-family share 0.278. Mean positives per language 2.95. Witnessed ambiguity 295. Novel-composition share 0.502 (210 templates whose AST shape is outside the procedural sampler).

Concept-level split, seed 2507: train 292, validation 63, held out 63. A family stays on one side. The evidence scorer and the expansion prior are not retrained on this bank.

Held-out families: `agriculture:flag`, `agriculture:soil_profile_horizon_designation`, `citation`, `cloud_infrastructure:bounded_numeric_identifier`, `cloud_infrastructure:network_interface_suffix`, `cloud_infrastructure:node_pool_ordinal`, `cloud_infrastructure:zone_suffix`, `document_processing:integer`, `document_processing:routing_label`, `education:assignment_revision_identifier`, `education:curriculum_document_revision_identifier`, `energy:contractual_gas_day_date`, and 44 more.

| Run | Prompt | Prompt domain | Accepted | Rejected |
| --- | --- | --- | ---: | ---: |
| 2026-09-29_terra_001 | real_world_concepts_v1.md | logistics | 6 | 0 |
| 2026-09-29_terra_002 | ambiguity_v1.md | logistics | 5 | 1 |
| 2026-09-29_terra_003 | adversarial_v1.md | logistics | 6 | 0 |
| 2026-09-29_terra_004 | domain_expansion_v1.md | software | 2 | 4 |
| 2026-09-29_terra_005 | domain_expansion_v1.md | legal | 3 | 3 |
| 2026-09-29_terra_006 | domain_expansion_v1.md | finance | 4 | 2 |
| 2026-09-29_terra_007 | domain_expansion_v1.md | healthcare | 3 | 3 |
| 2026-09-29_terra_008 | prior_decisions_v1.md | software | 7 | 1 |
| 2026-09-29_terra_009 | prior_decisions_v1.md | finance | 7 | 1 |
| 2026-09-29_terra_010 | prior_decisions_v1.md | legal | 7 | 1 |
| 2026-09-29_terra_011 | prior_decisions_v1.md | healthcare | 7 | 1 |
| 2026-09-29_terra_012 | prior_decisions_v1.md | logistics | 6 | 2 |
| 2026-09-29_terra_013 | prior_decisions_v1.md | manufacturing | 8 | 0 |
| 2026-09-29_terra_014 | prior_decisions_v1.md | energy | 7 | 1 |
| 2026-09-29_terra_015 | prior_decisions_v1.md | agriculture | 8 | 0 |
| 2026-09-29_terra_016 | prior_decisions_v1.md | retail | 7 | 1 |
| 2026-09-29_terra_017 | prior_decisions_v1.md | telecom | 7 | 1 |
| 2026-09-29_terra_018 | prior_decisions_v1.md | government | 8 | 0 |
| 2026-09-29_terra_019 | prior_decisions_v1.md | scientific_data | 8 | 0 |
| 2026-09-29_terra_020 | prior_decisions_v1.md | document_processing | 8 | 0 |
| 2026-09-29_terra_021 | prior_decisions_v1.md | cloud_infrastructure | 8 | 0 |
| 2026-09-29_terra_022 | prior_decisions_v1.md | education | 7 | 1 |
| 2026-09-29_terra_023 | prior_decisions_v1.md | insurance | 8 | 0 |
| 2026-09-29_terra_024 | real_world_concepts_v2.md | software | 6 | 1 |
| 2026-09-29_terra_025 | real_world_concepts_v2.md | finance | 7 | 1 |
| 2026-09-29_terra_026 | real_world_concepts_v2.md | legal | 8 | 0 |
| 2026-09-29_terra_027 | real_world_concepts_v2.md | healthcare | 8 | 0 |
| 2026-09-29_terra_028 | real_world_concepts_v2.md | logistics | 8 | 0 |
| 2026-09-29_terra_029 | real_world_concepts_v2.md | manufacturing | 7 | 1 |
| 2026-09-29_terra_030 | real_world_concepts_v2.md | energy | 8 | 0 |
| 2026-09-29_terra_031 | real_world_concepts_v2.md | agriculture | 7 | 0 |
| 2026-09-29_terra_032 | real_world_concepts_v2.md | retail | 8 | 0 |
| 2026-09-29_terra_033 | real_world_concepts_v2.md | telecom | 7 | 0 |
| 2026-09-29_terra_034 | real_world_concepts_v2.md | government | 5 | 3 |
| 2026-09-29_terra_035 | real_world_concepts_v2.md | scientific_data | 8 | 0 |
| 2026-09-29_terra_036 | real_world_concepts_v2.md | document_processing | 7 | 1 |
| 2026-09-29_terra_037 | real_world_concepts_v2.md | cloud_infrastructure | 8 | 0 |
| 2026-09-29_terra_038 | real_world_concepts_v2.md | education | 7 | 1 |
| 2026-09-29_terra_039 | real_world_concepts_v2.md | insurance | 7 | 0 |
| 2026-09-29_terra_040 | ambiguity_v2.md | software | 8 | 0 |
| 2026-09-29_terra_041 | ambiguity_v2.md | finance | 7 | 1 |
| 2026-09-29_terra_042 | ambiguity_v2.md | legal | 7 | 0 |
| 2026-09-29_terra_043 | ambiguity_v2.md | healthcare | 7 | 1 |
| 2026-09-29_terra_044 | ambiguity_v2.md | logistics | 7 | 1 |
| 2026-09-29_terra_045 | ambiguity_v2.md | manufacturing | 7 | 1 |
| 2026-09-29_terra_046 | ambiguity_v2.md | energy | 6 | 2 |
| 2026-09-29_terra_047 | ambiguity_v2.md | agriculture | 6 | 2 |
| 2026-09-29_terra_048 | ambiguity_v2.md | retail | 6 | 2 |
| 2026-09-29_terra_049 | ambiguity_v2.md | telecom | 6 | 2 |
| 2026-09-29_terra_050 | ambiguity_v2.md | government | 6 | 2 |
| 2026-09-29_terra_051 | ambiguity_v2.md | scientific_data | 6 | 2 |
| 2026-09-29_terra_052 | ambiguity_v2.md | document_processing | 5 | 3 |
| 2026-09-29_terra_053 | ambiguity_v2.md | cloud_infrastructure | 8 | 0 |
| 2026-09-29_terra_054 | ambiguity_v2.md | education | 7 | 0 |
| 2026-09-29_terra_055 | ambiguity_v2.md | insurance | 8 | 0 |
| 2026-09-29_terra_056 | domain_expansion_v1.md | manufacturing | 4 | 2 |
| 2026-09-29_terra_057 | domain_expansion_v1.md | energy | 5 | 1 |
| 2026-09-29_terra_058 | domain_expansion_v1.md | agriculture | 5 | 1 |
| 2026-09-29_terra_059 | domain_expansion_v1.md | retail | 4 | 2 |
| 2026-09-29_terra_060 | domain_expansion_v1.md | telecom | 5 | 1 |
| 2026-09-29_terra_061 | domain_expansion_v1.md | government | 5 | 1 |
| 2026-09-29_terra_062 | domain_expansion_v1.md | scientific_data | 5 | 1 |
| 2026-09-29_terra_063 | domain_expansion_v1.md | document_processing | 4 | 2 |
| 2026-09-29_terra_064 | domain_expansion_v1.md | cloud_infrastructure | 3 | 3 |
| 2026-09-29_terra_065 | domain_expansion_v1.md | education | 4 | 2 |
| 2026-09-29_terra_066 | domain_expansion_v1.md | insurance | 4 | 2 |

Calls 66. Spent $3.0687 of the $10 factory cap.

| Domain | Templates |
| --- | ---: |
| agriculture | 26 |
| air_freight | 1 |
| cloud_infrastructure | 27 |
| commercial_invoice | 1 |
| container_shipping | 1 |
| delivery_appointment_references | 1 |
| document_processing | 24 |
| education | 25 |
| energy | 26 |
| expedited_load_references | 1 |
| finance | 25 |
| freight_invoice_currency_fields | 1 |
| freight_rating | 1 |
| government | 24 |
| healthcare | 25 |
| insurance | 27 |
| legal | 25 |
| logistics | 28 |
| manufacturing | 26 |
| package_seal_identifiers | 1 |
| retail | 25 |
| scientific_data | 27 |
| shipping_documentation | 1 |
| software | 23 |
| telecom | 25 |
| warehouse_slot_labels | 1 |

| Rejection bucket | Count |
| --- | ---: |
| bad teacher output | 34 |
| DSL limitation | 23 |
| parser limitation | 17 |

| Bucket | Detail | Count |
| --- | --- | ---: |
| bad teacher output | positive_rejected | 9 |
| bad teacher output | duplicate_example | 6 |
| bad teacher output | memorized_literal | 6 |
| parser limitation | no_concepts | 6 |
| DSL limitation | unbound_literal | 5 |
| bad teacher output | bad_examples | 5 |
| DSL limitation | bad_width | 4 |
| DSL limitation | bad_literal | 4 |
| bad teacher output | near_miss_accepted | 4 |
| DSL limitation | bad_bounds | 4 |
| bad teacher output | too_few_positives | 4 |
| parser limitation | bad_repeat | 4 |
| DSL limitation | max_length | 3 |
| parser limitation | bad_opt | 3 |
| parser limitation | bad_int | 3 |
| DSL limitation | leading_zero | 2 |
| DSL limitation | repeat_cap | 1 |
| parser limitation | bad_seq | 1 |

| Reject reason | Count |
| --- | ---: |
| positive_rejected | 11 |
| bad_examples | 8 |
| duplicate_example | 6 |
| memorized_literal | 6 |
| no_concepts | 6 |
| unbound_literal | 5 |
| bad_width | 4 |
| bad_literal | 4 |
| near_miss_accepted | 4 |
| bad_bounds | 4 |
| too_few_positives | 4 |
| bad_repeat | 4 |
| bad_opt | 3 |
| bad_int | 3 |
| repeat_cap | 1 |
| bad_seq | 1 |

BENCH-E stays sealed. `needle()` is unchanged.
