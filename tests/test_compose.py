import random
import unittest

import torch

from pattern_induction.compose import (
    KIND_CLS,
    KIND_DELIM,
    KIND_LIT,
    KIND_OPT,
    KIND_REPEAT,
    TRAIN_FAMILIES,
    RepeatNum,
    build_split,
    rename_annotated,
    sample_batch,
    sample_episode,
)
from pattern_induction.membership import HELD_OUT
from pattern_induction.renaming import DIGIT, PUNCT, featurize
from pattern_induction.span_net import (
    SPAN_EQ_BACK_CLASS,
    SPAN_EQ_BACK_LEN,
    SPAN_IS_PAIR,
    SpanNet,
    collate,
    composition_loss,
    parameter_count,
    propose_spans,
)


def _seps(text: str, sep: str) -> int:
    return text.count(sep)


class OperatorLabelTest(unittest.TestCase):
    def test_training_families_stay_atomic_and_inside_the_alphabet(self):
        rng = random.Random(0)
        for family in TRAIN_FAMILIES:
            for label in (0, 1):
                episode = sample_episode(family, rng, label)
                self.assertEqual(episode.family, family)
                self.assertEqual(episode.label, label)
                blob = "".join(episode.support) + episode.query
                self.assertFalse(any(ch in HELD_OUT for ch in blob))

    def test_repeat_training_stops_at_three_and_extrapolation_continues(self):
        rng = random.Random(1)
        for _ in range(12):
            episode = sample_episode("repeat", rng, 1)
            sep = episode.pattern.split("sep=")[1]
            self.assertLessEqual(max(_seps(text, sep) for text in episode.support), 3)
            self.assertLessEqual(_seps(episode.query, sep), 3)
            self.assertGreaterEqual(_seps(episode.query, sep), 1)
        extra = sample_episode("repeat_extra", rng, 1)
        sep = extra.pattern.split("sep=")[1]
        self.assertLessEqual(max(_seps(text, sep) for text in extra.support), 3)
        self.assertGreaterEqual(_seps(extra.query, sep), 4)
        self.assertLessEqual(_seps(extra.query, sep), 5)
        broken = sample_episode("repeat_extra", rng, 0)
        self.assertEqual(broken.label, 0)

    def test_repeat_groups_share_a_span_and_mark_the_separator(self):
        lang = RepeatNum(random.Random(2))
        lang.sep = "."
        text = "12.4.8"
        self.assertTrue(lang.accepts(text))
        ann = lang.annotate(text)
        self.assertEqual(len(ann), len(text))
        self.assertEqual(ann.span[:2], [0, 0])
        self.assertEqual(ann.kind[:2], [KIND_CLS, KIND_CLS])
        self.assertEqual(ann.span[2:4], [1, 1])
        self.assertEqual(ann.kind[2], KIND_DELIM)
        self.assertEqual(ann.kind[3], KIND_REPEAT)
        self.assertEqual(ann.start[2], 1)
        self.assertEqual(ann.role[2][0], 1)
        self.assertEqual(ann.role[0][0], 0)
        self.assertNotEqual(ann.span[4], ann.span[2])

    def test_optional_tail_and_internal_group(self):
        tail = sample_episode("optional", random.Random(3), 1)
        seen_hyphen = False
        seen_plain = False
        for text, ann in zip(tail.support, tail.support_ann):
            self.assertEqual(ann.kind[0], KIND_LIT)
            if "-" in text:
                seen_hyphen = True
                index = text.index("-")
                self.assertEqual(ann.kind[index], KIND_OPT)
                self.assertEqual(ann.role[index][1], 1)
                self.assertEqual(ann.span[index], ann.span[0] + 1)
            else:
                seen_plain = True
                self.assertNotIn(KIND_OPT, ann.kind)
        self.assertTrue(seen_hyphen and seen_plain)

        internal = sample_episode("optional_internal", random.Random(4), 1)
        forms = set()
        for text, ann in zip(internal.support, internal.support_ann):
            forms.add(len(text))
            self.assertEqual(ann.kind[0], KIND_LIT)
            self.assertEqual(ann.kind[-1], KIND_LIT)
            if len(set(ann.span)) == 3:
                self.assertIn(KIND_OPT, ann.kind)
        self.assertGreater(len(forms), 1)

    def test_alternation_support_contains_both_branches(self):
        episode = sample_episode("alternation", random.Random(5), 1)
        kinds = []
        for ann in episode.support_ann:
            self.assertEqual(ann.role[0][2], 1)
            kinds.append(ann.kind[0])
        self.assertIn(KIND_LIT, kinds)
        self.assertIn(KIND_CLS, kinds)

    def test_held_out_compositions_are_not_a_training_family(self):
        rng = random.Random(6)
        batch = sample_batch("primitive", rng, 8)
        self.assertTrue(all(episode.family == "primitive" for episode in batch))
        for family in ("prefix_repeat", "repeat_optional", "prefix_alt"):
            episode = sample_episode(family, rng, 1)
            self.assertNotIn(episode.family, TRAIN_FAMILIES)
            self.assertEqual(sum(episode.label for episode in build_split(family, rng, 6)), 3)

    def test_rename_keeps_annotations_and_relational_features(self):
        episode = sample_episode("optional", random.Random(7), 1)
        renamed = rename_annotated(episode, random.Random(8), True)
        blob = "".join(renamed.support) + renamed.query
        self.assertTrue(any(ch in HELD_OUT for ch in blob))
        self.assertEqual([len(ann) for ann in episode.support_ann], [len(text) for text in renamed.support])
        self.assertEqual(episode.support_ann[0].span, renamed.support_ann[0].span)
        self.assertEqual(episode.query_ann.kind, renamed.query_ann.kind)
        original = featurize([episode.as_episode()], max_len=32, max_support=4)
        shifted = featurize([renamed.as_episode()], max_len=32, max_support=4)
        for left, right in zip(original, shifted):
            self.assertTrue(torch.equal(left, right))


class SpanModelTest(unittest.TestCase):
    def test_repeat_runs_match_two_back(self):
        classes = torch.tensor([[[DIGIT, PUNCT, DIGIT, PUNCT, DIGIT]]])
        pad = torch.zeros(1, 1, 5, dtype=torch.bool)
        _start, _end, feat, span_pad, _pos = propose_spans(classes, pad, max_spans=16)
        self.assertFalse(bool(span_pad[0, 0, 2]))
        self.assertEqual(float(feat[0, 0, 2, SPAN_IS_PAIR]), 0.0)
        self.assertEqual(float(feat[0, 0, 2, SPAN_EQ_BACK_CLASS]), 1.0)
        self.assertEqual(float(feat[0, 0, 2, SPAN_EQ_BACK_LEN]), 1.0)
        pair_at = int((feat[0, 0, :, SPAN_IS_PAIR] > 0.5).nonzero()[0])
        self.assertGreaterEqual(pair_at, 5)

    def test_capacity_invariance_and_one_step(self):
        model = SpanNet()
        count = parameter_count(model)
        self.assertGreaterEqual(count, 300_000)
        self.assertLessEqual(count, 2_000_000)
        names = [name for name, _param in model.named_parameters()]
        self.assertFalse(any("byte" in name or "codepoint" in name for name in names))
        episode = sample_episode("repeat", random.Random(9), 1)
        other = sample_episode("repeat", random.Random(10), 0)
        renamed = [
            rename_annotated(episode, random.Random(11)),
            rename_annotated(other, random.Random(12)),
        ]
        model.eval()
        first = collate([episode, other], max_len=model.max_len, max_support=model.max_support)
        second = collate(renamed, max_len=model.max_len, max_support=model.max_support)
        for left, right in zip(first[:4], second[:4]):
            self.assertTrue(torch.equal(left, right))
        with torch.no_grad():
            logit_a, _aux = model(*first[:4])
            logit_b, _aux = model(*second[:4])
        self.assertTrue(torch.allclose(logit_a, logit_b))
        model.train()
        batch = collate([episode, other], max_len=model.max_len, max_support=model.max_support)
        logit, aux = model(*batch[:4])
        loss, _mem, _aux_loss = composition_loss(logit, aux, *batch[4:])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(batch[9].sum()), 0)


if __name__ == "__main__":
    unittest.main()
