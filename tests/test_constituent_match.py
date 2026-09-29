import unittest

import torch

from pattern_induction.renaming import apply_mapping
from prior.constituent_match import (
    SUPPORT_FACT_DIM,
    ConstituentMatcher,
    facts_from_supports,
    lexicon_bit,
)
from prior.features import INDEX


class ConstituentMatchTest(unittest.TestCase):
    def test_facts_keep_a_tight_window_and_a_lexicon_column(self):
        tight = facts_from_supports([["615-616-617", "615-617-616", "616-615-617", "617-616-615"]])
        spread = facts_from_supports([["192-847-305", "760-118-904", "333-581-270", "409-662-155"]])
        self.assertEqual(tuple(tight.shape), (1, SUPPORT_FACT_DIM))
        self.assertLess(float(tight[0, 0]), float(spread[0, 0]))
        self.assertEqual(lexicon_bit(["USD-ab", "EUR-ab", "GBP-ab", "JPY-ab"]), 1.0)
        self.assertEqual(lexicon_bit(["AB-1234", "CD-5678", "EF-9012", "GH-3456"]), 0.0)

    def test_a_letter_rename_leaves_the_constituents_in_place(self):
        groups = [["ABC2024", "ABC2025", "ABC2026", "ABC2024"]]
        mapping = {"A": "X", "B": "Y", "C": "L"}
        renamed = [[apply_mapping(text, mapping) for text in groups[0]]]
        self.assertTrue(torch.allclose(facts_from_supports(groups), facts_from_supports(renamed), atol=1e-5))

    def test_the_interval_query_sees_the_digit_range(self):
        model = ConstituentMatcher()
        support = torch.zeros(2, SUPPORT_FACT_DIM)
        support[0, 0] = 0.01
        support[1, 0] = 0.9
        candidate = torch.zeros(2, 1, len(INDEX))
        candidate[:, :, INDEX["int_present"]] = 1.0
        scores = model(support, candidate)
        self.assertEqual(tuple(scores.shape), (2, 1))
        self.assertFalse(torch.allclose(scores[0], scores[1]))


if __name__ == "__main__":
    unittest.main()
