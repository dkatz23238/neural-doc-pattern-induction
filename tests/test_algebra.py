import random
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from pattern_induction.algebra import (
    AltOp,
    OperatorOracle,
    OptionalOp,
    RepeatOp,
    SeqOp,
    describe,
    operator_loss,
    schema_from_episode,
)
from pattern_induction.compose import rename_annotated, sample_episode
from pattern_induction.recipes import recipe_sources, sample_recipe_batch
from pattern_induction.span_net import SpanNet, collate, parameter_count


SHAPES = {
    "repeat": "REP2-4(L)",
    "optional": "SEQ(L,OPT(L),L)",
    "alternation": "ALT(L,L)",
    "prefix_repeat": "SEQ(L,REP2-4(L))",
    "repeat_optional": "SEQ(REP2-4(L),OPT(L))",
    "prefix_alt": "SEQ(L,ALT(L,L))",
}


class OperatorAlgebraTest(unittest.TestCase):
    def test_one_shared_module_per_operator(self):
        model = OperatorOracle()
        count = parameter_count(model)
        self.assertEqual(parameter_count(SpanNet()), 945464)
        self.assertGreaterEqual(count, 1_000_000)
        self.assertLessEqual(count, 2_000_000)
        for cls in (SeqOp, RepeatOp, OptionalOp, AltOp):
            self.assertEqual(sum(isinstance(module, cls) for module in model.modules()), 1)

    def test_gold_trees_cover_positive_queries(self):
        model = OperatorOracle()
        model.eval()
        for family, shape in SHAPES.items():
            for seed in range(4):
                episode = sample_episode(family, random.Random(seed), 1)
                self.assertEqual(describe(schema_from_episode(episode)), shape, family)
                packed = collate([episode], max_len=model.max_len, max_support=model.max_support)
                with torch.no_grad():
                    model(*packed[:4], [episode])
                cursor, length = model.last_cursors[0]
                self.assertGreaterEqual(cursor, length, family)

    def test_repeat_depth_uses_schema_bounds(self):
        model = OperatorOracle()
        model.eval()
        for seed in range(4):
            episode = sample_episode("repeat_extra", random.Random(seed), 1)
            separators = sum(episode.query.count(mark) for mark in "-_./")
            self.assertGreaterEqual(separators, 4)
            schema = schema_from_episode(episode)
            self.assertLessEqual(schema["max_n"], 4)
            self.assertGreaterEqual(schema["min_n"], 2)
            packed = collate([episode], max_len=model.max_len, max_support=model.max_support)
            with torch.no_grad():
                model(*packed[:4], [episode])
            self.assertLessEqual(model.last_repeat_max, 4)
            self.assertGreaterEqual(model.last_cursors[0][0], model.last_cursors[0][1])

    def test_query_annotation_does_not_change_the_logit(self):
        model = OperatorOracle()
        model.eval()
        episode = sample_episode("prefix_repeat", random.Random(3), 1)
        blank = replace(episode, query_ann=None)
        first = collate([episode], max_len=model.max_len, max_support=model.max_support)
        second = collate([blank], max_len=model.max_len, max_support=model.max_support)
        with torch.no_grad():
            left = model(*first[:4], [episode])
            right = model(*second[:4], [blank])
        self.assertEqual(float((left - right).abs()), 0.0)

    def test_eval_rename_is_invariant(self):
        model = OperatorOracle()
        model.eval()
        episodes = [sample_episode("repeat_optional", random.Random(index), index % 2) for index in range(4)]
        renamed = [rename_annotated(episode, random.Random(8000 + index), True) for index, episode in enumerate(episodes)]
        original = collate(episodes, max_len=model.max_len, max_support=model.max_support)
        shifted = collate(renamed, max_len=model.max_len, max_support=model.max_support)
        with torch.no_grad():
            left = model(*original[:4], episodes)
            right = model(*shifted[:4], renamed)
        self.assertTrue(torch.allclose(left, right, atol=1e-5))

    def test_compiled_schema_keeps_the_repeat(self):
        bank, _exposes = recipe_sources(Path(__file__).resolve().parents[1])["d1"]
        episode = sample_recipe_batch("d1", random.Random(0), 2, 2, bank)[0]
        self.assertIn("REP", describe(schema_from_episode(episode)))
        model = OperatorOracle()
        model.train()
        batch = sample_recipe_batch("d1", random.Random(1), 2, 4, bank)
        packed = collate(batch, max_len=model.max_len, max_support=model.max_support)
        logit = model(*packed[:4], batch)
        loss, _membership = operator_loss(logit, packed[4])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_atomic_batch_backward(self):
        model = OperatorOracle()
        model.train()
        batch = sample_recipe_batch("d0", random.Random(0), 1, 4, None)
        packed = collate(batch, max_len=model.max_len, max_support=model.max_support)
        logit = model(*packed[:4], batch)
        loss, _membership = operator_loss(logit, packed[4])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
