import unittest

from pattern_induction.selector import (
    FEATURE_NAMES,
    apply_gate,
    fit_listwise,
    pool_matrix,
    probability_gap,
    rank_index,
    raw_features,
    softmax,
)


class FeatureTest(unittest.TestCase):
    def test_alternation_literals_are_not_free_in_the_description(self):
        closed = raw_features(r"^(?:YES|NO)$", r"^[A-Z]{2,3}$", ["YES", "NO", "YES", "NO"])
        broad = raw_features(r"^[A-Z]{2,3}$", r"^[A-Z]{2,3}$", ["YES", "NO", "YES", "NO"])
        self.assertGreater(closed["alt_literal_chars"], 0.0)
        self.assertEqual(broad["alt_literal_chars"], 0.0)
        self.assertGreater(closed["n_alternations"], 0.0)
        self.assertEqual(len(pool_matrix([r"^(?:YES|NO)$", r"^[A-Z]{2,3}$"], r"^[A-Z]{2,3}$", ["YES", "NO"])[0]), len(FEATURE_NAMES))

    def test_features_use_only_the_visible_strings(self):
        strings = ["INV-1000", "INV-2044", "INV-3391"]
        first = raw_features(r"^INV-\d{4}$", r"^INV-\d{4}$", strings)
        second = raw_features(r"^INV-\d{4}$", r"^INV-\d{4}$", list(strings))
        self.assertEqual(first, second)
        self.assertEqual(first["is_heuristic"], 1.0)


class ListwiseTest(unittest.TestCase):
    def test_soft_labels_prefer_the_higher_accuracy(self):
        peaked = softmax([1.0 / 0.05, 0.2 / 0.05])
        self.assertGreater(peaked[0], 0.9)

    def test_a_separable_list_is_ranked_correctly(self):
        # One feature: the good candidate scores higher on it.
        train_rows = [[[1.0], [0.0]] for _ in range(8)]
        train_acc = [[1.0, 0.0] for _ in range(8)]
        patterns = [["good", "bad"] for _ in range(8)]
        heuristic = ["bad"] * 8
        weights, accuracy, _step = fit_listwise(
            train_rows,
            train_acc,
            train_rows,
            train_acc,
            patterns,
            heuristic,
            lam=0.01,
            tau=0.1,
            steps=30,
            lr=0.5,
            patience=10,
        )
        self.assertGreater(accuracy, 0.9)
        scores = [weights[0] * 1.0, weights[0] * 0.0]
        self.assertEqual(rank_index(scores, ["good", "bad"], "bad"), 0)

    def test_gate_keeps_the_baseline_without_a_margin(self):
        self.assertEqual(apply_gate("learned", "rules", prob_gap=0.01, threshold=0.2), "rules")
        self.assertEqual(apply_gate("learned", "rules", prob_gap=0.5, threshold=0.2), "learned")
        self.assertGreater(probability_gap([5.0, 0.0]), probability_gap([0.1, 0.0]))


if __name__ == "__main__":
    unittest.main()
