import random
import unittest

import torch

from pattern_induction.equivariant_net import (
    EquivariantNet,
    collate_episodes,
    equivariant_loss,
    parameter_count,
)
from pattern_induction.membership import Episode
from pattern_induction.renaming import (
    HELD_OUT,
    PUNCT,
    REL_LEFT_CONSTANT,
    REL_LEFT_ID,
    REL_NEW,
    featurize,
    rename_episode,
    structural_class,
)


class RenameFeatureTest(unittest.TestCase):
    def test_separator_class_ignores_the_codepoint(self):
        self.assertEqual(structural_class("-"), PUNCT)
        self.assertEqual(structural_class("§"), PUNCT)
        self.assertEqual(structural_class("~"), PUNCT)
        self.assertNotEqual(structural_class("A"), structural_class("1"))

    def test_rename_preserves_relations_and_introduces_a_held_out_mark(self):
        episode = Episode(["AB-12", "AB-34", "AB-56"], "AB-90", 1, "^x$", "t", "positive")
        renamed = rename_episode(episode, random.Random(0), force_holdout=True)
        blob = "".join(renamed.support) + renamed.query
        self.assertTrue(any(ch in HELD_OUT for ch in blob))
        self.assertNotEqual(blob, "".join(episode.support) + episode.query)
        original = featurize([episode])
        shifted = featurize([renamed])
        for left, right in zip(original, shifted):
            self.assertTrue(torch.equal(left, right))

    def test_constant_prefix_matches_and_a_new_digit_does_not(self):
        episode = Episode(["AB-12", "AB-34"], "AB-90", 1, "", "t", "positive")
        _local, _classes, rel, pad = featurize([episode])
        query = -1
        self.assertFalse(bool(pad[0, query, 0]))
        self.assertEqual(float(rel[0, query, 0, REL_LEFT_CONSTANT]), 1.0)
        self.assertEqual(float(rel[0, query, 0, REL_LEFT_ID]), 1.0)
        self.assertEqual(float(rel[0, query, 0, REL_NEW]), 0.0)
        self.assertEqual(float(rel[0, query, 2, REL_LEFT_CONSTANT]), 1.0)
        self.assertEqual(float(rel[0, query, 3, REL_LEFT_CONSTANT]), 0.0)
        self.assertEqual(float(rel[0, query, 3, REL_NEW]), 1.0)

    def test_wrong_separator_is_new_in_a_constant_column(self):
        episode = Episode(["AB-12", "AB-34"], "AB.90", 0, "", "t", "edit")
        _local, _classes, rel, _pad = featurize([episode])
        self.assertEqual(float(rel[0, -1, 2, REL_NEW]), 1.0)
        self.assertEqual(float(rel[0, -1, 2, REL_LEFT_ID]), 0.0)
        self.assertEqual(float(rel[0, -1, 2, REL_LEFT_CONSTANT]), 1.0)


class EquivariantModelTest(unittest.TestCase):
    def test_capacity_and_one_step(self):
        model = EquivariantNet()
        count = parameter_count(model)
        self.assertGreaterEqual(count, 300_000)
        self.assertLessEqual(count, 2_000_000)
        names = {name for name, _param in model.named_parameters()}
        self.assertFalse(any("byte" in name for name in names))
        episode = Episode(["AB-12", "AB-34"], "AB-90", 1, "", "t", "positive")
        other = Episode(["AB-12", "AB-34"], "AX-90", 0, "", "t", "edit")
        renamed = [
            rename_episode(episode, random.Random(1)),
            rename_episode(other, random.Random(2)),
        ]
        first = collate_episodes([episode, other])
        second = collate_episodes(renamed)
        for left, right in zip(first[:4], second[:4]):
            self.assertTrue(torch.equal(left, right))
        logit, latent = model(*first[:4])
        logit2, latent2 = model(*second[:4])
        loss, _bce, _agree = equivariant_loss(logit, latent, logit2, latent2, first[4])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tuple(logit.shape), (2,))
        self.assertEqual(latent.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
