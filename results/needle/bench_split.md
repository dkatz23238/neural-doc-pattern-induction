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
