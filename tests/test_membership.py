import random
import re
import unittest

import torch

from pattern_induction.membership import (
    HELD_OUT,
    auroc,
    build_split,
    make_episode,
    sample_batch,
    structural_negative,
)
from pattern_induction.membership_net import (
    POINTER_SCALE,
    MembershipNet,
    collate,
    membership_loss,
    parameter_count,
)
from pattern_induction.schema import compile_parts, sample_task


class DataTest(unittest.TestCase):
    def test_training_strings_hide_held_out_bytes(self):
        rng = random.Random(0)
        for _ in range(20):
            episode = make_episode(rng, False, _ % 2, 4, 0.0, False, "train")
            self.assertIsNotNone(episode)
            blob = "".join(episode.support) + episode.query
            for ch in HELD_OUT:
                self.assertNotIn(ch, blob)
            compiled = re.compile(episode.pattern)
            if episode.label:
                self.assertIsNotNone(compiled.fullmatch(episode.query))
                self.assertEqual(len(episode.invariant), len(episode.query))
            else:
                self.assertIsNone(compiled.fullmatch(episode.query))

    def test_shifted_split_uses_a_held_out_byte(self):
        episodes = build_split(random.Random(1), 12, True, 4, 0.0, True, "ood")
        self.assertGreaterEqual(len(episodes), 16)
        self.assertTrue(any(episode.held_out() for episode in episodes))
        for episode in episodes:
            compiled = re.compile(episode.pattern)
            matched = compiled.fullmatch(episode.query) is not None
            self.assertEqual(matched, bool(episode.label))

    def test_curriculum_pairs_respect_the_holdout(self):
        rng = random.Random(5)
        batch = sample_batch(rng, 8, level=0, structural_rate=0.0, easy_rate=0.0)
        self.assertEqual(len(batch), 8)
        for episode in batch:
            blob = "".join(episode.support) + episode.query
            for ch in HELD_OUT:
                self.assertNotIn(ch, blob)
            compiled = re.compile(episode.pattern)
            matched = compiled.fullmatch(episode.query) is not None
            self.assertEqual(matched, bool(episode.label))

    def test_structural_negative_is_rejected_by_gold(self):
        rng = random.Random(2)
        found = False
        for _ in range(8):
            _parts, support, pattern = sample_task(rng, 5, shift=False)
            compiled = re.compile(pattern)
            text = structural_negative(support, pattern, compiled, rng, False)
            if text is None:
                continue
            found = True
            self.assertIsNone(compiled.fullmatch(text))
            self.assertTrue(all(ch not in HELD_OUT for ch in text))
        self.assertTrue(found)


class ModelTest(unittest.TestCase):
    def test_capacity_and_frozen_identity(self):
        model = MembershipNet()
        count = parameter_count(model)
        self.assertGreaterEqual(count, 1_000_000)
        self.assertLessEqual(count, 5_000_000)
        names = {name for name, _param in model.named_parameters()}
        self.assertNotIn("identity", names)
        self.assertNotIn("id_proj", names)
        self.assertTrue(torch.equal(model.identity[ord("A")], model.identity[ord("A")]))
        self.assertFalse(torch.equal(model.identity[ord("A")], model.identity[ord("#")]))

    def test_one_step_and_auroc(self):
        torch.manual_seed(0)
        model = MembershipNet()
        batch = sample_batch(random.Random(3), 4, structural_rate=0.0)
        tokens, pad, labels, invariant = collate(batch)
        logit, inv_logit = model(tokens, pad)
        self.assertEqual(tuple(logit.shape), (4,))
        loss, _bce, _aux = membership_loss(logit, inv_logit, labels, invariant)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 0.99)
        self.assertLess(auroc([0, 1], [0.9, 0.1]), 0.01)
        self.assertAlmostEqual(auroc([0, 1, 0, 1], [0.4, 0.4, 0.4, 0.4]), 0.5)

    def test_pointer_scale_peaks_on_the_matching_byte(self):
        logits = torch.zeros(6 * 48)
        logits[0] = POINTER_SCALE
        weight = torch.softmax(logits, dim=0)
        self.assertGreater(float(weight[0]), 0.9)

    def test_schema_samples_still_compile(self):
        pattern = compile_parts(sample_task(random.Random(4), 3, shift=True)[0])
        self.assertIsNotNone(pattern)


if __name__ == "__main__":
    unittest.main()
