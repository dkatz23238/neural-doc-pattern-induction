import math
import unittest

import torch

from prior.expansion import DIMS, HEADS, MAX_NODES, ExpansionPrior, parameter_count
from prior.typed_lambda import (
    INIT_LAMBDA,
    TypedLambdaPrior,
    seed_passes,
    seed_retains,
    selection_score,
    slice_mask,
)


def _blank(count: int = 2) -> dict:
    batch = {
        "mask": torch.ones(count, 32, dtype=torch.bool),
        "evidence": torch.zeros(count, 32),
    }
    for name in HEADS:
        batch[f"{name}_feat"] = torch.zeros(count, 32, MAX_NODES, DIMS[name])
        batch[f"{name}_expand"] = torch.zeros(count, 32, MAX_NODES)
        batch[f"{name}_mask"] = torch.zeros(count, 32, MAX_NODES)
    return batch


def _report(held_top: float, evidence_top: float, slice_combined: float, slice_evidence: float) -> dict:
    slices = {
        name: {"combined": slice_combined, "evidence": slice_evidence, "n": 10} for name in (
            "charset_expansion",
            "open_width",
            "integer_expansion",
            "loose_repeat",
        )
    }
    return {
        "factory_heldout": {"top1": held_top, "evidence_top1": evidence_top, "n": 20},
        "heldout_slices": slices,
        "val_slices": slices,
        "ood_slices": slices,
        "val_by_family": {"width": 0.70},
        "val_evidence_by_family": {"width": 0.70},
        "ood_by_family": {"repeat_interval": 0.50},
        "ood_evidence_by_family": {"repeat_interval": 0.50},
    }


class TypedLambdaTests(unittest.TestCase):
    def test_each_scale_starts_near_zero_and_the_shared_prior_stays_at_one(self):
        typed = TypedLambdaPrior()
        shared = ExpansionPrior()
        self.assertEqual(parameter_count(shared), 1765)
        self.assertEqual(parameter_count(typed), 1768)
        self.assertAlmostEqual(float(shared.scale().detach()), 1.0, places=5)
        for name, value in typed.scales().items():
            self.assertAlmostEqual(float(value.detach()), INIT_LAMBDA, places=4, msg=name)

    def test_a_scale_near_zero_leaves_the_program_with_evidence(self):
        model = TypedLambdaPrior()
        batch = _blank()
        batch["width_expand"][:, 0, 0] = 1
        batch["width_mask"][:, 0, 0] = 1
        batch["evidence"][:, 0] = 3

        def fake(name, feat, expand):
            return torch.ones_like(expand) if name == "width" else torch.zeros_like(expand)

        model.scores_from = fake
        model.raw_lambda.data.fill_(math.log(math.expm1(1e-6)))
        scores = model.combined_scores(batch)
        self.assertTrue(torch.allclose(scores, batch["evidence"], atol=1e-4))
        model.raw_lambda.data[HEADS.index("width")] = math.log(math.expm1(1.0))
        moved = model.combined_scores(batch)
        self.assertTrue(torch.allclose(moved[:, 0], torch.full((2,), 4.0), atol=1e-4))
        self.assertTrue(torch.allclose(moved[:, 1:], torch.zeros(2, 31), atol=1e-4))

    def test_only_the_active_scale_receives_program_gradient(self):
        model = TypedLambdaPrior()
        model.raw_lambda.data.fill_(math.log(math.expm1(1.0)))
        batch = _blank()
        batch["integer_expand"][:, 0, 0] = 1
        batch["integer_mask"][:, 0, 0] = 1

        def fake(name, feat, expand):
            return torch.ones_like(expand) if name == "integer" else torch.zeros_like(expand)

        model.scores_from = fake
        model.combined_scores(batch).sum().backward()
        grad = model.raw_lambda.grad
        self.assertGreater(float(grad[HEADS.index("integer")].abs()), 0)
        for name in ("charset", "width", "repeat"):
            self.assertAlmostEqual(float(grad[HEADS.index(name)]), 0.0, places=5)

    def test_selection_score_weights_the_two_validation_sets_equally(self):
        self.assertAlmostEqual(selection_score(0.2, 0.8), 0.5)

    def test_slice_mask_requires_a_gold_expansion_and_a_rival(self):
        batch = {
            "integer_hyp_gold": torch.zeros(2, 4),
            "integer_hyp_mask": torch.zeros(2, 4),
            "integer_hyp_expand": torch.zeros(2, 4),
        }
        batch["integer_hyp_gold"][0, 0] = 1
        batch["integer_hyp_mask"][0, 0] = 1
        batch["integer_hyp_expand"][0, 0] = 1
        batch["integer_hyp_mask"][0, 1] = 1
        chosen = slice_mask(batch, "integer_expansion")
        self.assertTrue(bool(chosen[0]))
        self.assertFalse(bool(chosen[1]))

    def test_gate_rejects_a_slice_under_evidence_and_keeps_a_match(self):
        even = _report(0.50, 0.50, 0.40, 0.40)
        self.assertTrue(seed_passes(even))
        self.assertFalse(seed_retains(even))
        short = _report(0.50, 0.50, 0.30, 0.40)
        short["heldout_slices"]["integer_expansion"] = {"combined": 0.30, "evidence": 0.40, "n": 10}
        self.assertFalse(seed_passes(short))
        gained = _report(0.55, 0.50, 0.50, 0.40)
        self.assertTrue(seed_passes(gained))
        self.assertTrue(seed_retains(gained))
        contaminated = _report(0.50, 0.50, 0.40, 0.40)
        contaminated["val_by_family"] = {"width": 0.60}
        contaminated["val_evidence_by_family"] = {"width": 0.70}
        self.assertFalse(seed_passes(contaminated))


if __name__ == "__main__":
    unittest.main()
