import random
import unittest

import torch

from pattern_induction.language import GrammarPrior, boundary_strings, group_languages
from pattern_induction.membership import Episode
from pattern_induction.program import Prog, Task, execute, feature_batch, sample_compose_task
from pattern_induction.renaming import rename_episode
from pattern_induction.scope import ScopeNet, total_parameter_count
from pattern_induction.span_net import SpanNet, parameter_count
from pattern_induction.witness_net import WitnessNet, collate_witnesses, geometry, witness_loss


def _optional_pair():
    supports = ["XKJY", "XK68JY"]
    optional = Prog(
        "SEQ",
        children=[
            Prog("COPY", slot=0, start=0, end=2),
            Prog("OPT", children=[Prog("CLS", cls="digit", lo=2, hi=2)]),
            Prog("COPY", slot=0, start=2, end=4),
        ],
    )
    present = Prog(
        "SEQ",
        children=[
            Prog("COPY", slot=1, start=0, end=2),
            Prog("CLS", cls="digit", lo=2, hi=2),
            Prog("COPY", slot=1, start=4, end=6),
        ],
    )
    absent = Prog("COPY", slot=0, start=0, end=4)
    alternation = Prog("ALT", children=[present, absent])
    return supports, optional, alternation


class LanguageSelectionTest(unittest.TestCase):
    def test_equivalent_languages_merge_and_repeat_bounds_stay_apart(self):
        supports, optional, alternation = _optional_pair()
        self.assertTrue(execute(optional, "XKJY", supports))
        self.assertTrue(execute(optional, "XK68JY", supports))
        self.assertTrue(execute(alternation, "XK12JY", supports))
        self.assertFalse(execute(optional, "XK7JY", supports))
        groups = group_languages([optional, alternation], supports)
        self.assertEqual(len(groups), 1)
        literal = Prog(
            "SEQ",
            children=[
                Prog("COPY", slot=0, start=0, end=2),
                Prog("OPT", children=[Prog("COPY", slot=1, start=2, end=4)]),
                Prog("COPY", slot=0, start=2, end=4),
            ],
        )
        self.assertIn("XK66JY", boundary_strings(supports))
        self.assertTrue(execute(optional, "XK66JY", supports))
        self.assertFalse(execute(literal, "XK66JY", supports))
        self.assertEqual(len(group_languages([optional, literal], supports)), 2)
        closed = Prog(
            "SEQ",
            children=[
                Prog("CLS", cls="digit", lo=1, hi=2),
                Prog(
                    "REPEAT",
                    lo=1,
                    hi=3,
                    children=[Prog("SEQ", children=[Prog("COPY", slot=0, start=1, end=2), Prog("CLS", cls="digit", lo=1, hi=2)])],
                ),
            ],
        )
        opened = Prog(
            "SEQ",
            children=[
                Prog("CLS", cls="digit", lo=1, hi=2),
                Prog(
                    "REPEAT",
                    lo=1,
                    hi=8,
                    children=[Prog("SEQ", children=[Prog("COPY", slot=0, start=1, end=2), Prog("CLS", cls="digit", lo=1, hi=2)])],
                ),
            ],
        )
        repeat_supports = ["6-38", "54-5-45"]
        self.assertEqual(len(group_languages([closed, opened], repeat_supports)), 2)
        prior = GrammarPrior()
        prior.observe(opened, repeat_bounds=True)
        self.assertGreater(prior.score(opened), prior.score(closed))

    def test_witness_head_is_small_and_finite(self):
        self.assertEqual(parameter_count(SpanNet()), 945464)
        scope = ScopeNet()
        witness = WitnessNet(scope.local)
        total = total_parameter_count(scope) + sum(param.numel() for param in witness.head.parameters())
        self.assertGreaterEqual(total, 1_000_000)
        self.assertLessEqual(total, 2_000_000)
        self.assertLess(sum(param.numel() for param in witness.head.parameters()), 200_000)
        task = sample_compose_task("repeat", random.Random(3), 1)
        texts = boundary_strings(task.supports)
        self.assertGreater(len(texts), 0)
        batch = collate_witnesses([(task.supports, texts[0]), (task.supports, texts[min(1, len(texts) - 1)])])
        target = torch.tensor([1.0, 0.0])
        loss = witness_loss(witness(batch), target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(witness.head[0].weight.grad)
        self.assertTrue(all(param.grad is None for param in scope.parameters()))

    def test_geometry_follows_a_rename(self):
        task = sample_compose_task("prefix_repeat", random.Random(5), 1)
        episode = Episode(task.supports, task.supports[0], 1, task.family, "select", task.family)
        renamed = rename_episode(episode, random.Random(6), True)
        left = feature_batch([task])
        right = feature_batch([Task(task.family, renamed.support, task.program, [])])
        self.assertTrue(all(torch.allclose(a, b) for a, b in zip(left, right)))
        query = task.supports[0]
        mapped = renamed.support[0]
        self.assertEqual(geometry(task.supports, query), geometry(renamed.support, mapped))


if __name__ == "__main__":
    unittest.main()
