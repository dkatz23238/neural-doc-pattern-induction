import random
import unittest

import torch

from pattern_induction.local_net import LocalNet, collate_local, hypotheses_from_batch, local_loss
from pattern_induction.membership import Episode
from pattern_induction.program import Task, feature_batch, sample_compose_task
from pattern_induction.renaming import rename_episode
from pattern_induction.span_net import SpanNet, parameter_count


class LocalRecognizerTest(unittest.TestCase):
    def test_capacity_supervision_and_a_finite_step(self):
        self.assertEqual(parameter_count(SpanNet()), 945464)
        model = LocalNet()
        count = parameter_count(model)
        self.assertGreaterEqual(count, 1_000_000)
        self.assertLessEqual(count, 2_000_000)
        tasks = [sample_compose_task(family, random.Random(family.__len__()), 1) for family in ("primitive", "repeat", "optional", "alternation")]
        batch = collate_local(tasks)
        self.assertGreater(float(batch["repeat"].sum()), 0.0)
        self.assertGreater(float(batch["optional"].sum()), 0.0)
        self.assertGreater(float(batch["pair_y"].sum()), 0.0)
        model.train()
        loss, _atom = local_loss(model(batch), batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_rename_keeps_local_hypotheses(self):
        model = LocalNet()
        model.eval()
        task = sample_compose_task("prefix_repeat", random.Random(1), 1)
        episode = Episode(task.supports, task.supports[0], 1, task.family, "local", task.family)
        renamed = rename_episode(episode, random.Random(2), True)
        left = feature_batch([task])
        right = feature_batch([Task(task.family, renamed.support, task.program, [])])
        self.assertTrue(all(torch.allclose(a, b) for a, b in zip(left, right)))
        a = hypotheses_from_batch(model, collate_local([task]), [task.supports], 4)[0]
        b = hypotheses_from_batch(model, collate_local([Task(task.family, renamed.support, task.program, [])]), [renamed.support], 4)[0]
        sig = lambda hyps: sorted((item.kind, item.slot, item.start, item.end, item.index) for item in hyps)
        self.assertEqual(sig(a), sig(b))
        self.assertGreater(len(a), 0)


if __name__ == "__main__":
    unittest.main()
