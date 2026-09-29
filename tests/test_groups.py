import math
import random
import unittest

import torch

from pattern_induction.compose import PrefixRepeat, RepeatNum, build_split, rename_annotated, sample_batch, sample_episode
from pattern_induction.group_net import GroupNet, batch_targets, hierarchy_loss
from pattern_induction.groups import (
    G_IS_PREFIX,
    G_IS_REPEAT,
    G_LOG_UNITS,
    OP_ALT,
    OP_OPTIONAL,
    OP_REPEAT,
    OP_SEQ,
    REL_SAME,
    lift_tree,
    propose_groups,
    propose_slot,
)
from pattern_induction.recipes import D1_SPECS
from pattern_induction.renaming import DIGIT, PUNCT, UPPER, featurize
from pattern_induction.schema_dsl import Language, compile_schema
from pattern_induction.span_net import SpanNet, collate, parameter_count


def _text_classes(text: str) -> list[int]:
    classes = []
    for ch in text:
        if ch.isupper():
            classes.append(UPPER)
        elif ch.isdigit():
            classes.append(DIGIT)
        else:
            classes.append(PUNCT)
    return classes


class TreeLiftTest(unittest.TestCase):
    def test_repeat_is_one_operator_at_training_depth_and_at_five(self):
        lang = RepeatNum(random.Random(0))
        lang.sep = "."
        for seps in (1, 3, 5):
            text = lang.sample(random.Random(seps), seps)
            tree = lift_tree(lang.annotate(text))
            repeats = [index for index, node in enumerate(tree) if node.op == OP_REPEAT]
            self.assertEqual(len(repeats), 1, text)
            units = [node for node in tree if node.parent == repeats[0]]
            self.assertEqual(len(units), seps + 1, text)
            self.assertEqual(len({node.unit for node in units}), 1)
            self.assertTrue(any(node.op == OP_SEQ and node.parent < 0 for node in tree))

    def test_prefix_stays_outside_the_repeat(self):
        lang = PrefixRepeat(random.Random(1))
        lang.prefix = "QX"
        lang.repeat.sep = "."
        text = "QX" + lang.repeat.sample(random.Random(2), 4)
        tree = lift_tree(lang.annotate(text))
        repeat = next(node for node in tree if node.op == OP_REPEAT)
        self.assertGreaterEqual(repeat.start, 2)
        self.assertTrue(any(node.op != OP_REPEAT and node.start == 0 and node.end == 2 for node in tree))

    def test_optional_and_alternation_are_operators(self):
        optional = sample_episode("optional", random.Random(2), 1)
        self.assertTrue(any(lift_tree(ann) and any(node.op == OP_OPTIONAL for node in lift_tree(ann)) for ann in optional.support_ann))
        alt = sample_episode("alternation", random.Random(3), 1)
        self.assertTrue(all(any(node.op == OP_ALT for node in lift_tree(ann)) for ann in alt.support_ann))
        prefixed = sample_episode("prefix_alt", random.Random(4), 1)
        tree = lift_tree(prefixed.query_ann)
        self.assertTrue(any(node.op == OP_ALT for node in tree))
        self.assertTrue(any(node.op == OP_SEQ and node.parent < 0 for node in tree))

    def test_schema_repeat_lifts(self):
        compiled = compile_schema(D1_SPECS[0], allow_heldout=False)
        lang = Language(compiled, random.Random(0))
        found = False
        for plan in lang.plans():
            _text, ann = lang.sample(random.Random(1), plan)
            if any(node.op == OP_REPEAT for node in lift_tree(ann)):
                found = True
                break
        self.assertTrue(found)


class ProposalTest(unittest.TestCase):
    def test_five_separators_stay_one_repeat_node(self):
        text = "1.22.3.4.5.16"
        nodes = propose_slot(_text_classes(text), [0.0] * len(text), [0.0] * len(text))
        repeats = [node for node in nodes if node["kind"] == "repeat"]
        self.assertEqual(len(repeats), 1)
        self.assertEqual(repeats[0]["units"], 6)
        self.assertEqual(repeats[0]["start"], 0)
        self.assertEqual(repeats[0]["end"], len(text))

    def test_constant_prefix_is_its_own_group(self):
        text = "AB12.3.45"
        left = [1.0, 1.0] + [0.0] * (len(text) - 2)
        nodes = propose_slot(_text_classes(text), left, [0.0] * len(text))
        prefix = [node for node in nodes if node["prefix"] == 1.0]
        self.assertEqual(len(prefix), 1)
        self.assertEqual((prefix[0]["start"], prefix[0]["end"]), (0, 2))
        repeat = next(node for node in nodes if node["kind"] == "repeat")
        self.assertEqual(repeat["start"], 2)
        self.assertGreaterEqual(repeat["units"], 2)

    def test_rename_preserves_the_graph_and_the_score(self):
        episode = sample_episode("prefix_repeat", random.Random(0), 1)
        renamed = rename_annotated(episode, random.Random(1), True)
        model = GroupNet("h2")
        model.eval()

        def packs(row):
            return featurize([row.as_episode()], max_len=model.max_len, max_support=model.max_support)

        left = propose_groups(*packs(episode)[1:])
        right = propose_groups(*packs(renamed)[1:])
        self.assertTrue(torch.equal(left["start"], right["start"]))
        self.assertTrue(torch.equal(left["end"], right["end"]))
        self.assertTrue(torch.equal(left["edge_rel"], right["edge_rel"]))
        with torch.no_grad():
            first, _aux = model(*packs(episode))
            second, _aux = model(*packs(renamed))
        self.assertTrue(torch.allclose(first, second, atol=1e-5))


class HierarchyModelTest(unittest.TestCase):
    def test_parameter_bands_and_a_finite_step(self):
        span = parameter_count(SpanNet())
        self.assertEqual(span, 945464)
        for arch in ("h1", "h2"):
            model = GroupNet(arch)
            count = parameter_count(model)
            self.assertGreaterEqual(count, 1_000_000, arch)
            self.assertLessEqual(count, 2_000_000, arch)
            batch = sample_batch("repeat", random.Random(0), 4)
            packed = collate(batch, max_len=model.max_len, max_support=model.max_support)
            logit, aux = model(*packed[:4])
            targets = batch_targets(batch, packed[1], packed[2], packed[3], model.max_len)
            query = model.max_support
            repeat_groups = targets["proposed"]["feat"][:, query, :, G_IS_REPEAT] > 0.5
            self.assertTrue(bool(repeat_groups.any()))
            edge = model.score_pairs(aux["groups"], targets["pair_src"], targets["pair_dst"])
            loss, _mem, _aux = hierarchy_loss(logit, aux, *packed[4:], targets, edge)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()

    def test_repeat_group_is_labeled_repeat(self):
        episode = sample_episode("repeat", random.Random(5), 1)
        packed = collate([episode], max_len=32, max_support=4)
        targets = batch_targets([episode], packed[1], packed[2], packed[3], 32)
        query = 4
        feat = targets["proposed"]["feat"][0, query]
        op = targets["op"][0, query]
        repeat_at = [index for index in range(feat.shape[0]) if float(feat[index, G_IS_REPEAT]) > 0.5]
        self.assertEqual(len(repeat_at), 1)
        self.assertEqual(int(op[repeat_at[0]]), OP_REPEAT)
        expected = math.log1p(float(episode.query.count(episode.pattern.split("sep=")[1]) + 1)) / math.log1p(8.0)
        self.assertAlmostEqual(float(feat[repeat_at[0], G_LOG_UNITS]), expected, places=4)
        prefix = sample_episode("prefix_repeat", random.Random(6), 1)
        packed = collate([prefix], max_len=32, max_support=4)
        proposed = propose_groups(packed[1], packed[2], packed[3])
        self.assertTrue(bool((proposed["feat"][0, 4, :, G_IS_PREFIX] > 0.5).any()))
        self.assertTrue(bool((proposed["edge_rel"][0] == REL_SAME).any()))

    def test_eval_split_still_builds(self):
        rows = build_split("repeat_extra", random.Random(5000), 4)
        self.assertEqual(len(rows), 4)
