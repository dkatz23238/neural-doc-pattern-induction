import unittest

import torch

from pattern_induction.override import (
    PAIR_FEATURES,
    ResidualOverride,
    choose_alternative,
    collate_pairs,
    override_loss,
    pair_rows,
)
from pattern_induction.override_split import composition_regexes, generate_split
from pattern_induction.ranker import encode_utf8


class SplitTest(unittest.TestCase):
    def test_train_and_val_regexes_are_disjoint(self):
        train = composition_regexes("train")
        val = composition_regexes("val")
        self.assertTrue(train)
        self.assertTrue(val)
        self.assertTrue(train.isdisjoint(val))
        self.assertTrue(any("ACK" in regex for regex in val))
        self.assertFalse(any("ACK" in regex for regex in train))

    def test_samples_match_their_gold_regex(self):
        tasks = generate_split(12, "train", seed=1) + generate_split(12, "val", seed=2)
        import re

        for task in tasks:
            gold = re.compile(task.gold_regex)
            for text in task.visible + task.hidden_positives:
                self.assertIsNotNone(gold.fullmatch(text), task.id)
            for text in task.hidden_negatives:
                self.assertIsNone(gold.fullmatch(text), task.id)


class DecisionTest(unittest.TestCase):
    def test_threshold_keeps_the_rule_until_the_gap_is_large_enough(self):
        self.assertIsNone(choose_alternative([0.04, -0.1], threshold=0.05))
        self.assertEqual(choose_alternative([0.04, 0.2], threshold=0.05), 1)
        self.assertIsNone(choose_alternative([], threshold=0.0))

    def test_false_override_weight_does_not_punish_a_pessimistic_gain(self):
        delta = torch.tensor([[0.2, -0.3]])
        logit = torch.tensor([[2.0, -2.0]])
        actual = torch.tensor([[0.5, -0.2]])
        valid = torch.tensor([[True, True]])
        low, _parts = override_loss(delta, logit, actual, valid, lam=1.0)
        high, _parts = override_loss(delta, logit, actual, valid, lam=8.0)
        self.assertGreater(float(high), float(low))
        gain = torch.tensor([[0.2]])
        gain_logit = torch.tensor([[1.0]])
        gain_actual = torch.tensor([[0.5]])
        gain_valid = torch.tensor([[True]])
        calm, _parts = override_loss(gain, gain_logit, gain_actual, gain_valid, lam=1.0)
        heavy, _parts = override_loss(gain, gain_logit, gain_actual, gain_valid, lam=8.0)
        self.assertAlmostEqual(float(calm), float(heavy), places=5)


class ModelTest(unittest.TestCase):
    def test_pair_features_are_visible_only(self):
        rows = pair_rows([r"^[A-Z]{2,3}$"], r"^(?:YES|NO)$", r"^(?:YES|NO)$", ["YES", "NO", "YES"])
        self.assertEqual(len(rows[0]), len(PAIR_FEATURES))
        again = pair_rows([r"^[A-Z]{2,3}$"], r"^(?:YES|NO)$", r"^(?:YES|NO)$", ["YES", "NO", "YES"])
        self.assertEqual(rows, again)

    def test_one_step_reduces_loss(self):
        torch.manual_seed(0)
        model = ResidualOverride(cross_attention=True, d=32, heads=4, string_layers=1, regex_layers=1, max_len=32)
        rows = []
        for index in range(8):
            strings = [f"AB-{index:04d}"]
            rule = r"^AB-\d{4}$"
            alt = rf"^AB-{index:04d}$"
            string_ids, string_pad, alt_ids, alt_pad = [], [], [], []
            for text in strings:
                ids, pad, _cut = encode_utf8(text, max_len=32)
                string_ids.append(ids)
                string_pad.append(pad)
            rule_ids, rule_pad, _cut = encode_utf8(rule, max_len=32)
            ids, pad, _cut = encode_utf8(alt, max_len=32)
            alt_ids.append(ids)
            alt_pad.append(pad)
            features = [[0.0] * len(PAIR_FEATURES)]
            features[0][0] = 1.0 if index % 2 == 0 else -1.0
            rows.append(
                {
                    "string_ids": string_ids,
                    "string_pad": string_pad,
                    "rule_ids": rule_ids,
                    "rule_pad": rule_pad,
                    "alt_ids": alt_ids,
                    "alt_pad": alt_pad,
                    "features": features,
                    "alt_delta": [0.4 if index % 2 == 0 else -0.2],
                }
            )
        mean = [0.0] * len(PAIR_FEATURES)
        scale = [1.0] * len(PAIR_FEATURES)
        batch = collate_pairs(rows, mean, scale, max_len=32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        model.train()
        delta, logit = model(
            batch["strings"],
            batch["string_pad"],
            batch["string_slot_pad"],
            batch["rule"],
            batch["rule_pad"],
            batch["alts"],
            batch["alt_pad"],
            batch["alt_slot_pad"],
            batch["features"],
        )
        first, _parts = override_loss(delta, logit, batch["actual"], ~batch["alt_slot_pad"], lam=5.0)
        first.backward()
        optimizer.step()
        delta, logit = model(
            batch["strings"],
            batch["string_pad"],
            batch["string_slot_pad"],
            batch["rule"],
            batch["rule_pad"],
            batch["alts"],
            batch["alt_pad"],
            batch["alt_slot_pad"],
            batch["features"],
        )
        second, _parts = override_loss(delta, logit, batch["actual"], ~batch["alt_slot_pad"], lam=5.0)
        self.assertLess(float(second.detach()), float(first.detach()))

    def test_pooled_forward_matches_cross_shapes(self):
        torch.manual_seed(1)
        rows = []
        strings = ["INV-1000", "INV-2044"]
        string_ids, string_pad = [], []
        for text in strings:
            ids, pad, _cut = encode_utf8(text, max_len=32)
            string_ids.append(ids)
            string_pad.append(pad)
        rule_ids, rule_pad, _cut = encode_utf8(r"^INV-\d{4}$", max_len=32)
        alt_ids, alt_pad = [], []
        for pattern in (r"^INV-\d+$", r"^.*$"):
            ids, pad, _cut = encode_utf8(pattern, max_len=32)
            alt_ids.append(ids)
            alt_pad.append(pad)
        rows.append(
            {
                "string_ids": string_ids,
                "string_pad": string_pad,
                "rule_ids": rule_ids,
                "rule_pad": rule_pad,
                "alt_ids": alt_ids,
                "alt_pad": alt_pad,
                "features": [[0.1] * len(PAIR_FEATURES), [-0.2] * len(PAIR_FEATURES)],
                "alt_delta": [0.1, -0.4],
            }
        )
        mean = [0.0] * len(PAIR_FEATURES)
        scale = [1.0] * len(PAIR_FEATURES)
        batch = collate_pairs(rows, mean, scale, max_len=32)
        for cross in (True, False):
            model = ResidualOverride(
                cross_attention=cross,
                d=32,
                heads=4,
                string_layers=1,
                regex_layers=1,
                max_len=32,
            )
            model.eval()
            delta, logit = model(
                batch["strings"],
                batch["string_pad"],
                batch["string_slot_pad"],
                batch["rule"],
                batch["rule_pad"],
                batch["alts"],
                batch["alt_pad"],
                batch["alt_slot_pad"],
                batch["features"],
            )
            self.assertEqual(tuple(delta.shape), (1, 2))
            self.assertEqual(tuple(logit.shape), (1, 2))
            self.assertTrue(torch.isfinite(delta).all())
            self.assertTrue(torch.isfinite(logit).all())


if __name__ == "__main__":
    unittest.main()
