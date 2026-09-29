import random
import unittest

import torch

from pattern_induction.local_train import _truncate
from pattern_induction.membership import Episode
from pattern_induction.program import Task, feature_batch, sample_compose_task
from pattern_induction.renaming import rename_episode
from pattern_induction.scope import (
    PAIR_NAMES,
    ScopeNet,
    collate_scope,
    hypotheses_from_scope,
    scope_loss,
    total_parameter_count,
)
from pattern_induction.span_net import SpanNet, parameter_count


class ScopeRelationTest(unittest.TestCase):
    def test_capacity_labels_and_a_finite_step(self):
        self.assertEqual(parameter_count(SpanNet()), 945464)
        model = ScopeNet()
        count = total_parameter_count(model)
        self.assertGreaterEqual(count, 1_000_000)
        self.assertLessEqual(count, 2_000_000)
        self.assertLess(parameter_count(model), count)
        families = {
            "repeat": "same_unit",
            "optional": "optional_child",
            "alternation": "alt_branch",
            "prefix_alt": "shared_prefix",
        }
        tasks = [sample_compose_task(family, random.Random(len(family) + 3), 1) for family in families]
        batch = collate_scope(tasks)
        for row, channel in enumerate(families.values()):
            if channel == "optional_child":
                self.assertGreater(float(batch["unary_y"][row, ..., 2].sum()), 0.0)
            else:
                self.assertGreater(float(batch["rel_y"][row, :, PAIR_NAMES.index(channel)].sum()), 0.0)
        model.local.eval()
        loss, _pair = scope_loss(model(batch), batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(param.grad is None for param in model.local.parameters()))
        self.assertIsNotNone(model.pair_out.weight.grad)
        self.assertTrue(torch.isfinite(model.pair_out.weight.grad).all())

    def test_rename_keeps_scope_hypotheses(self):
        model = ScopeNet()
        model.local.eval()
        task = sample_compose_task("prefix_repeat", random.Random(1), 1)
        episode = Episode(task.supports, task.supports[0], 1, task.family, "scope", task.family)
        renamed = rename_episode(episode, random.Random(2), True)
        left = feature_batch([task])
        right = feature_batch([Task(task.family, renamed.support, task.program, [])])
        self.assertTrue(all(torch.allclose(a, b) for a, b in zip(left, right)))
        a = hypotheses_from_scope(model, collate_scope([task]), [task.supports], 4)[0]
        b = hypotheses_from_scope(model, collate_scope([Task(task.family, renamed.support, task.program, [])]), [renamed.support], 4)[0]
        signature = lambda hyps: sorted(
            (item.kind, item.slot, item.start, item.end, item.cls, item.lo, item.hi, item.slot2, item.start2, item.end2, item.index)
            for item in hyps
        )
        self.assertEqual(signature(a), signature(b))
        self.assertGreater(len(a), 0)
        self.assertEqual(len(_truncate(a, 8)), len(a))

    def test_second_bound_is_kept_for_a_class_span(self):
        model = ScopeNet()
        model.local.eval()
        task = sample_compose_task("repeat", random.Random(4), 1)
        hyps = hypotheses_from_scope(model, collate_scope([task]), [task.supports], 8)[0]
        grouped: dict[tuple, set[tuple[int, int]]] = {}
        for hyp in hyps:
            if hyp.kind != "cls":
                continue
            grouped.setdefault((hyp.slot, hyp.start, hyp.end, hyp.cls), set()).add((hyp.lo, hyp.hi))
        self.assertTrue(any(len(bounds) >= 2 for bounds in grouped.values()))


if __name__ == "__main__":
    unittest.main()
