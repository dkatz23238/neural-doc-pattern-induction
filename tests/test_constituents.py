import random
import unittest
from pathlib import Path

import torch

from pattern_induction.membership import Episode
from pattern_induction.program import MAX_LEN, MAX_SUPPORT
from pattern_induction.renaming import DIGIT, N_CLASS, UPPER, apply_mapping, build_mapping, featurize
from prior.constituents import (
    FIELD_FEAT,
    N_POS,
    N_SLOTS,
    RUN_FEAT,
    STRUCTURE_DIM,
    digit_constancy as digit_channels,
    encode_groups,
    field_matrix,
    last_right_constant,
    numeric_coverage,
    structure_matrix,
    suffix_role,
)
from prior.support_encode import load_frozen_encoder


def _structure(groups: list[list[str]]) -> torch.Tensor:
    episodes = [
        Episode(list(group[:MAX_SUPPORT]), group[0], 1, "prior", "prior", "prior") for group in groups
    ]
    _local, classes, rel, pad = featurize(episodes, max_len=MAX_LEN, max_support=MAX_SUPPORT)
    return structure_matrix(classes, pad, rel)


def _at(row: torch.Tensor, slot: int, pos: int) -> torch.Tensor:
    start = (slot * N_POS + pos) * RUN_FEAT
    return row[start : start + RUN_FEAT]


class ConstituentTest(unittest.TestCase):
    def test_an_optional_prefix_stays_a_different_run_from_the_field(self):
        row = _structure([["AB-1234", "5678", "AB-9012", "3456"]])[0]
        long_head = _at(row, 0, 0)
        short_head = _at(row, 1, 0)
        self.assertEqual(int(long_head[UPPER].item()), 1)
        self.assertEqual(int(short_head[DIGIT].item()), 1)
        self.assertEqual(int(short_head[UPPER].item()), 0)
        self.assertEqual(row.numel(), STRUCTURE_DIM)

    def test_a_repeated_field_matches_the_run_two_back(self):
        row = _structure([["12-34-56", "78-90", "11-22-33", "44-55-66"]])[0]
        # Run 0 digit, run 1 punct, run 2 digit. The third position is run 2.
        third = _at(row, 0, 2)
        self.assertEqual(int(third[DIGIT].item()), 1)
        # one-hot (5), length (1), eq prev class/len (2), eq back class/len (2)
        self.assertEqual(int(third[N_CLASS + 1 + 2].item()), 1)
        self.assertEqual(int(third[N_CLASS + 1 + 3].item()), 1)

    def test_letter_rename_does_not_change_the_run_layout(self):
        groups = [["AB-1234", "5678", "AB-9012", "3456"]]
        mapping = build_mapping(groups[0], random.Random(1), False)
        renamed = [[apply_mapping(text, mapping) for text in groups[0]]]
        self.assertTrue(torch.equal(_structure(groups), _structure(renamed)))

    def test_a_repeated_interval_shares_field_columns(self):
        interval = field_matrix([["615-616-617", "615-617-616", "616-615-617", "617-616-615"]])[0]
        width = field_matrix([["12-34-56", "78-90-12", "11-22-33", "44-55-66"]])[0]
        # Digit constancy is the third field feature of each run.
        def digit_constancy(row):
            values = []
            for slot in range(N_SLOTS):
                for pos in range(N_POS):
                    start = (slot * N_POS + pos) * FIELD_FEAT
                    values.append(float(row[start + 2]))
            return max(values)

        self.assertGreater(digit_constancy(interval), 0.5)
        self.assertGreater(digit_constancy(interval), digit_constancy(width) + 0.3)

    def test_frozen_span_state_follows_the_same_runs(self):
        encoder = load_frozen_encoder(Path("."))
        _structure_feat, neural = encode_groups(
            encoder,
            [["12-34-56", "78-90-12", "11-22-33", "44-55-66"], ["AB-1234", "5678", "AB-9012", "3456"]],
        )
        self.assertEqual(tuple(neural.shape), (2, N_SLOTS * N_POS * 160))
        first_run = neural[0, :160]
        self.assertGreater(float(first_run.abs().sum()), 0.0)
        # The short support's first run is the digit field, a different constituent from the prefix.
        short = neural[1, 160 : 320]
        long = neural[1, :160]
        self.assertGreater(float((short - long).abs().sum()), 0.0)

    def test_local_slices_keep_the_digit_channel_and_the_suffix_role(self):
        field = field_matrix([["615-616-617", "615-617-616", "616-615-617", "617-616-615"]])
        channels = digit_channels(field)
        self.assertEqual(tuple(channels.shape), (1, N_SLOTS * N_POS))
        self.assertGreater(float(channels.max()), 0.5)
        structure = _structure([["USD-ab", "EUR-ab", "GBP-ab", "JPY-ab"]])
        role = suffix_role(structure)
        self.assertEqual(tuple(role.shape), (1, N_SLOTS * (RUN_FEAT - N_CLASS)))
        right = last_right_constant(structure)
        self.assertGreater(float(right[0, 0]), 0.9)

    def test_a_repeated_window_is_a_tighter_digit_constituent_than_a_spread(self):
        tight = numeric_coverage([["615-616-617", "618-619-620", "621-622-623", "624-625-626"]])
        spread = numeric_coverage([["192-847-305", "760-118-904", "333-581-270", "409-662-155"]])
        bare = numeric_coverage([["6152", "6158", "6155", "6159"]])
        letters = numeric_coverage([["USD-ab", "EUR-ab", "GBP-ab", "JPY-ab"]])
        self.assertLess(float(tight), 0.05)
        self.assertLess(float(bare), 0.05)
        self.assertGreater(float(spread), 0.3)
        self.assertEqual(float(letters), 1.0)


if __name__ == "__main__":
    unittest.main()
