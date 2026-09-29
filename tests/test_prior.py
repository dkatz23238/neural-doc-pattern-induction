import random
import unittest

import torch

from pattern_induction.program import Prog
from prior.config import TrainConfig, load_config
from prior.features import FEATURE_DIM, candidate_features
from prior.loss import ranking_loss
from prior.factors import AFFIX, FEAT_DIM, FIELD, OPTIONAL, REPEAT, describe, pack_candidates
from prior.model import (
    PriorModel,
    ast_parameter_count,
    choose_ast,
    choose_factor,
    choose_width,
    factor_parameter_count,
    trainable_parameters,
)
from prior.support_encode import SUPPORT_DIM


class PriorTest(unittest.TestCase):
    def test_sizes_stay_near_their_targets(self):
        for target in (10_000, 30_000, 100_000, 300_000, 1_000_000):
            width, hidden = choose_width(target)
            model = PriorModel(TrainConfig(kind="mlp", target_params=target))
            count = trainable_parameters(model)
            self.assertEqual((model.width, model.hidden), (width, hidden))
            self.assertLess(abs(count - target) / target, 0.15)

    def test_symbolic_prior_has_no_parameters(self):
        model = PriorModel(TrainConfig(kind="symbolic", target_params=0))
        self.assertEqual(trainable_parameters(model), 0)
        batch = {
            "support": torch.zeros(2, SUPPORT_DIM),
            "candidate": torch.zeros(2, 3, FEATURE_DIM),
            "mask": torch.ones(2, 3, dtype=torch.bool),
            "gold": torch.zeros(2, 3, dtype=torch.bool),
        }
        batch["candidate"][:, 1, 0] = 1
        scores = model(batch)
        self.assertEqual(tuple(scores.shape), (2, 3))

    def test_convention_and_hull_are_different_features(self):
        supports = ["2024", "2026", "2025", "2024"]
        hull = candidate_features(Prog("INT", spec="2024:2026", lo=4, hi=4), supports)
        wide = candidate_features(Prog("INT", spec="2000:2099", lo=4, hi=4), supports)
        self.assertNotEqual(hull, wide)
        self.assertEqual(len(hull), FEATURE_DIM)

    def test_listwise_loss_prefers_the_latent_candidate(self):
        gold = torch.tensor([[False, True, False]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        peaked = ranking_loss(torch.tensor([[0.0, 4.0, 0.0]]), gold, mask)
        flat = ranking_loss(torch.zeros(1, 3), gold, mask)
        self.assertLess(float(peaked), float(flat))

    def test_a_tiny_mlp_learns_which_candidate_is_marked(self):
        model = PriorModel(TrainConfig(kind="mlp", target_params=10_000, seed=0))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
        support = torch.randn(32, SUPPORT_DIM)
        candidate = torch.randn(32, 4, FEATURE_DIM) * 0.01
        candidate[:, 2, :] += 1
        gold = torch.zeros(32, 4, dtype=torch.bool)
        gold[:, 2] = True
        mask = torch.ones(32, 4, dtype=torch.bool)
        batch = {"support": support, "candidate": candidate, "mask": mask, "gold": gold}
        for _ in range(60):
            optimizer.zero_grad()
            loss = ranking_loss(model(batch), gold, mask)
            loss.backward()
            optimizer.step()
        choice = model(batch).argmax(dim=-1)
        self.assertGreater(float((choice == 2).float().mean()), 0.9)

    def test_compositional_holdout_is_disjoint_and_each_family_builds(self):
        from prior.generate import OOD_FAMILIES, TRAIN_FAMILIES, build_task

        self.assertTrue(set(OOD_FAMILIES).isdisjoint(TRAIN_FAMILIES))
        for name in ("prefix_width", "charset_width", "prefix_optional", "repeat_width"):
            self.assertIn(name, TRAIN_FAMILIES)
        for name in ("prefix_charset_optional", "repeat_interval", "lexicon_suffix"):
            self.assertIn(name, OOD_FAMILIES)
        rng = random.Random(2507)
        for family in dict.fromkeys((*TRAIN_FAMILIES, *OOD_FAMILIES)):
            task = build_task(rng, family)
            self.assertIsNotNone(task, family)
            self.assertEqual(task["family"], family)
            self.assertTrue(task["signature"])
            self.assertGreater(sum(task["gold"]), 0)
            from prior.tree import pack_program, programs_from_json

            for prog in programs_from_json(task["programs"]):
                pack_program(prog, task["supports"])

    def test_ast_prior_stays_in_its_band(self):
        bands = ((15_000, 10_000, 20_000), (40_000, 30_000, 50_000), (90_000, 80_000, 100_000))
        for target, low, high in bands:
            width, hidden, tree_dim = choose_ast(target)
            model = PriorModel(TrainConfig(kind="ast", target_params=target, name="prior_ast"))
            count = trainable_parameters(model)
            self.assertEqual((model.width, model.hidden, model.tree_dim), (width, hidden, tree_dim))
            self.assertEqual(count, ast_parameter_count(width, hidden, tree_dim))
            self.assertGreaterEqual(count, low)
            self.assertLessEqual(count, high)
            self.assertGreaterEqual(model.tree_dim, 32)
            self.assertLessEqual(model.tree_dim, 64)

    def test_repeat_is_one_function_for_a_class_and_a_range(self):
        model = PriorModel(TrainConfig(kind="ast", target_params=15_000, name="prior_ast"))
        charset = Prog("REPEAT", lo=1, hi=3, children=[Prog("CLS", cls="upper", lo=2, hi=2)])
        interval = Prog("REPEAT", lo=1, hi=2, children=[Prog("INT", spec="10:99", lo=2, hi=2)])
        from prior.tree import pack_candidates

        left = pack_candidates([charset], ["AB", "CD", "EF", "GH"], 1)
        right = pack_candidates([interval], ["21", "34", "40", "55"], 1)
        keys = ("kind", "params", "left", "right", "depth", "root", "relation")
        batch = {key: torch.stack([left[key], right[key]]) for key in keys}
        batch["support"] = torch.zeros(2, SUPPORT_DIM)
        scores = model(batch)
        scores.sum().backward()
        weight = model.encoder.repeat.net[0].weight
        self.assertIsNotNone(weight.grad)
        self.assertGreater(float(weight.grad.abs().sum()), 0.0)
        self.assertFalse(torch.allclose(scores[0], scores[1], atol=1e-6))

    def test_letter_rename_does_not_change_the_tree(self):
        program = Prog(
            "SEQ",
            children=[
                Prog("COPY", slot=0, start=0, end=3),
                Prog("INT", spec="2000:2099", lo=4, hi=4),
            ],
        )
        supports = ["ABC2024", "ABC2025", "ABC2026", "ABC2024"]
        renamed = ["XYZ2024", "XYZ2025", "XYZ2026", "XYZ2024"]
        from prior.tree import pack_program

        left = pack_program(program, supports)
        right = pack_program(program, renamed)
        for key in left:
            self.assertTrue(torch.equal(left[key], right[key]), key)
        model = PriorModel(TrainConfig(kind="ast", target_params=15_000, name="prior_ast"))
        from prior.tree import pack_candidates

        packed = [pack_candidates([program], group, 1) for group in (supports, renamed)]
        keys = ("kind", "params", "left", "right", "depth", "root", "relation")
        batch = {key: torch.stack([item[key] for item in packed]) for key in keys}
        batch["support"] = torch.zeros(2, SUPPORT_DIM)
        scores = model(batch)
        self.assertTrue(torch.allclose(scores[0], scores[1], atol=1e-5))

    def test_factor_prior_stays_in_its_band(self):
        bands = ((15_000, 10_000, 20_000), (40_000, 30_000, 50_000))
        for target, low, high in bands:
            hidden = choose_factor(target)
            model = PriorModel(TrainConfig(kind="factor", target_params=target, name="prior_factor"))
            count = trainable_parameters(model)
            self.assertEqual(model.width, hidden)
            self.assertEqual(count, factor_parameter_count(hidden))
            self.assertGreaterEqual(count, low)
            self.assertLessEqual(count, high)
            self.assertEqual(model.factor.proj[REPEAT].in_features, FEAT_DIM)

    def test_repeat_slot_ignores_the_child(self):
        supports = ["12-34", "56-78-90", "12-34-56", "78-90"]

        def wrapped(field: Prog) -> Prog:
            inner = Prog(field.op, cls=field.cls, lo=field.lo, hi=field.hi, spec=field.spec)
            return Prog(
                "SEQ",
                children=[
                    field,
                    Prog(
                        "REPEAT",
                        lo=1,
                        hi=3,
                        children=[Prog("SEQ", children=[Prog("COPY", slot=0, start=2, end=3), inner])],
                    ),
                ],
            )

        digit = wrapped(Prog("CLS", cls="digit", lo=2, hi=2))
        interval = wrapped(Prog("INT", spec="10:99", lo=2, hi=2))
        left, left_mask, _left_sig = describe(digit, supports)
        right, right_mask, _right_sig = describe(interval, supports)
        self.assertEqual(left[REPEAT], right[REPEAT])
        self.assertEqual(left_mask[REPEAT], right_mask[REPEAT])
        self.assertNotEqual(left[FIELD], right[FIELD])
        optional_digit = Prog(
            "SEQ",
            children=[Prog("OPT", children=[Prog("COPY", slot=0, start=0, end=1)]), Prog("CLS", cls="digit", lo=2, hi=2)],
        )
        optional_set = Prog(
            "SEQ",
            children=[
                Prog("OPT", children=[Prog("COPY", slot=0, start=0, end=1)]),
                Prog("SET", spec="digit", lo=2, hi=2),
            ],
        )
        opt_left, _, _ = describe(optional_digit, ["A12", "34", "B56", "78"])
        opt_right, _, _ = describe(optional_set, ["A12", "34", "B56", "78"])
        self.assertEqual(opt_left[OPTIONAL], opt_right[OPTIONAL])
        self.assertEqual(opt_left[AFFIX], opt_right[AFFIX])
        self.assertNotEqual(opt_left[FIELD], opt_right[FIELD])

    def test_a_wrong_field_can_still_have_the_right_repeat(self):
        supports = ["12-34", "56-78-90", "12-34-56", "78-90"]
        gold = Prog(
            "SEQ",
            children=[
                Prog("CLS", cls="digit", lo=2, hi=2),
                Prog(
                    "REPEAT",
                    lo=1,
                    hi=3,
                    children=[
                        Prog(
                            "SEQ",
                            children=[
                                Prog("COPY", slot=0, start=2, end=3),
                                Prog("CLS", cls="digit", lo=2, hi=2),
                            ],
                        )
                    ],
                ),
            ],
        )
        wrong = Prog(
            "SEQ",
            children=[
                Prog("CLS", cls="upper", lo=2, hi=2),
                Prog(
                    "REPEAT",
                    lo=1,
                    hi=3,
                    children=[
                        Prog(
                            "SEQ",
                            children=[
                                Prog("COPY", slot=0, start=2, end=3),
                                Prog("CLS", cls="upper", lo=2, hi=2),
                            ],
                        )
                    ],
                ),
            ],
        )
        packed = pack_candidates([gold, wrong], supports, [True, False], 2)
        self.assertEqual(packed["aux"][0, 0].tolist(), [1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertEqual(packed["aux"][0, 1, FIELD].item(), 0.0)
        self.assertEqual(packed["aux"][0, 1, REPEAT].item(), 1.0)
        self.assertEqual(packed["aux"][0, 1, AFFIX].item(), 1.0)

    def test_letter_rename_does_not_change_the_factors(self):
        program = Prog(
            "SEQ",
            children=[
                Prog("COPY", slot=0, start=0, end=3),
                Prog("INT", spec="2000:2099", lo=4, hi=4),
            ],
        )
        supports = ["ABC2024", "ABC2025", "ABC2026", "ABC2024"]
        renamed = ["XYZ2024", "XYZ2025", "XYZ2026", "XYZ2024"]
        left = pack_candidates([program], supports, [True], 1)
        right = pack_candidates([program], renamed, [True], 1)
        for key in left:
            self.assertTrue(torch.equal(left[key], right[key]), key)
        model = PriorModel(TrainConfig(kind="factor", target_params=15_000, name="prior_factor"))
        batch = {
            "support": torch.zeros(2, SUPPORT_DIM),
            "feat": torch.cat([left["feat"], right["feat"]]),
            "present": torch.cat([left["present"], right["present"]]),
        }
        scores = model(batch)
        self.assertTrue(torch.allclose(scores[0], scores[1], atol=1e-5))

    def test_interaction_cannot_exceed_one(self):
        program = Prog(
            "SEQ",
            children=[
                Prog("CLS", cls="digit", lo=2, hi=2),
                Prog("REPEAT", lo=1, hi=3, children=[Prog("CLS", cls="digit", lo=2, hi=2)]),
            ],
        )
        packed = pack_candidates([program], ["12", "1234", "123456", "12"], [True], 1)
        model = PriorModel(TrainConfig(kind="factor", target_params=15_000, name="prior_factor"))
        for layer in model.factor.proj:
            torch.nn.init.constant_(layer.weight, 0.1)
            torch.nn.init.constant_(layer.bias, 0.1)
        torch.nn.init.constant_(model.factor.support.weight, 0.1)
        torch.nn.init.constant_(model.factor.support.bias, 0.1)
        batch = {
            "support": torch.ones(1, SUPPORT_DIM),
            "feat": packed["feat"],
            "present": packed["present"],
        }
        model.factor.beta_repeat.data.fill_(0)
        model.factor.beta_optional.data.fill_(0)
        with torch.no_grad():
            base = model(batch)
            model.factor.beta_repeat.data.fill_(10)
            shifted = model(batch)
        delta = (shifted - base).abs()
        self.assertLessEqual(float(delta.max()), 1.0 + 1e-4)
        self.assertGreater(float(delta.max()), 0.01)

    def test_config_overrides_are_the_only_device_switch(self):
        cfg = load_config("configs/prior_100k.yaml", ["device=cpu", "model_size=30k", "steps=5"])
        self.assertEqual(cfg.device, "cpu")
        self.assertEqual(cfg.target_params, 30_000)
        self.assertEqual(cfg.steps, 5)
        self.assertEqual(cfg.kind, "mlp")


if __name__ == "__main__":
    unittest.main()
