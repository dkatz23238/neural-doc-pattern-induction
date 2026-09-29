import unittest

import torch

from pattern_induction.program import Prog
from prior.config import TrainConfig, load_config
from prior.features import FEATURE_DIM, candidate_features
from prior.loss import ranking_loss
from prior.model import PriorModel, choose_width, trainable_parameters
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

    def test_config_overrides_are_the_only_device_switch(self):
        cfg = load_config("configs/prior_100k.yaml", ["device=cpu", "model_size=30k", "steps=5"])
        self.assertEqual(cfg.device, "cpu")
        self.assertEqual(cfg.target_params, 30_000)
        self.assertEqual(cfg.steps, 5)
        self.assertEqual(cfg.kind, "mlp")


if __name__ == "__main__":
    unittest.main()
