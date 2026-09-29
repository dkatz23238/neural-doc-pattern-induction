import unittest

import torch

from pattern_induction.safe_override import (
    EASY_MARGIN,
    FEATURE_NAMES,
    LAMBDA,
    SafeOverride,
    assemble_features,
    choose_threshold,
    clears_minimum,
    clears_target,
    eligible,
    override_choice,
    parameter_count,
    safe_loss,
    summarize,
)


def _row(symbolic: float, alternative: float, failure: str, disagreed: bool = True) -> dict:
    return {
        "symbolic_accuracy": symbolic,
        "alt_accuracy": alternative,
        "failure": failure,
        "disagreed": disagreed,
    }


class SafeOverrideTest(unittest.TestCase):
    def test_default_keeps_the_symbolic_winner(self) -> None:
        self.assertFalse(override_choice(0.99, 0.99, True))
        self.assertFalse(override_choice(0.99, 0.5, False))
        self.assertTrue(override_choice(0.91, 0.9, True))

    def test_wrong_overrides_are_penalized_more(self) -> None:
        logit = torch.zeros(2)
        label = torch.tensor([1.0, 0.0])
        loss = safe_loss(logit, label, lam=LAMBDA)
        negative = safe_loss(logit[1:], label[1:], lam=LAMBDA)
        positive = safe_loss(logit[:1], label[:1], lam=LAMBDA)
        self.assertAlmostEqual(float(negative), float(positive) * LAMBDA, places=5)
        self.assertGreater(float(loss), float(positive))

    def test_threshold_refuses_an_easy_family_regression(self) -> None:
        rows = [
            _row(0.99, 0.90, "success"),
            _row(0.99, 0.90, "success"),
            _row(0.50, 0.90, "too_broad"),
        ]
        # High probability on every row: a low threshold overrides the easy family.
        probabilities = [0.8, 0.8, 0.8]
        threshold, info = choose_threshold(rows, probabilities)
        self.assertGreaterEqual(threshold, 0.8)
        chosen = info["chosen"]
        self.assertGreaterEqual(chosen["easy_accuracy"], chosen["easy_symbolic"] - EASY_MARGIN)
        self.assertTrue(chosen["eligible"] or threshold == 1.0)

    def test_eligible_requires_the_symbolic_floor(self) -> None:
        summary = summarize(
            [_row(0.96, 0.90, "success"), _row(0.96, 0.99, "too_broad")],
            [True, True],
        )
        self.assertLess(summary["semantic_accuracy"], summary["symbolic_accuracy"])
        self.assertFalse(eligible(summary))
        self.assertFalse(clears_minimum(summary))
        kept = summarize(
            [_row(0.96, 0.90, "success"), _row(0.96, 0.972, "too_broad")],
            [False, True],
        )
        self.assertGreater(kept["net"], 0)
        self.assertTrue(clears_minimum(kept))
        self.assertFalse(clears_target(kept))

    def test_model_stays_small_and_features_line_up(self) -> None:
        model = SafeOverride()
        self.assertLess(parameter_count(model), 5000)
        vector = assemble_features("^\\d{4}$", "^\\d{2}$", ["12", "1234"], 0.4, True, 0.7)
        self.assertEqual(len(vector), len(FEATURE_NAMES))
        self.assertEqual(vector[-3:], [0.4, 1.0, 0.7])


if __name__ == "__main__":
    unittest.main()
