import json
import random
import tempfile
import unittest
from pathlib import Path

import torch

from pattern_induction.compose import TRAIN_FAMILIES, sample_batch
from pattern_induction.recipes import d1_bank, sample_recipe_batch
from pattern_induction.renaming import featurize, rename_episode
from pattern_induction.schema_dsl import (
    Language,
    SchemaError,
    compile_schema,
    languages_differ,
    sample_schema_episode,
)
from pattern_induction.teacher import estimate_cost, extract_payload, reserve


PREFIX_REPEAT = {
    "id": "heldout_prefix_repeat",
    "schema": {
        "type": "seq",
        "children": [
            {"type": "literal_slot", "role": "prefix", "class": "upper", "min": 2, "max": 2},
            {"type": "class", "class": "digit", "min": 1, "max": 2},
            {
                "type": "repeat",
                "min": 1,
                "max": 3,
                "child": {
                    "type": "seq",
                    "children": [
                        {"type": "literal_slot", "role": "sep", "class": "punct", "min": 1, "max": 1},
                        {"type": "class", "class": "digit", "min": 1, "max": 1},
                    ],
                },
            },
        ],
    },
}


class DslTest(unittest.TestCase):
    def test_d1_matrix_holds_out_the_eval_languages(self):
        bank = d1_bank()
        self.assertEqual(len(bank), 8)
        self.assertTrue(all(item.heldout is None for item in bank))
        rng = random.Random(2)
        for compiled in bank:
            for label in (1, 0):
                episode = sample_schema_episode(compiled, rng, label, compiled.ident)
                self.assertEqual(episode.label, label)
                self.assertGreaterEqual(len(episode.support), 3)

    def test_prefix_repeat_is_rejected_until_a_recipe_allows_it(self):
        with self.assertRaises(SchemaError) as caught:
            compile_schema(PREFIX_REPEAT, allow_heldout=False)
        self.assertIn("prefix_repeat", caught.exception.reason)
        compiled = compile_schema(PREFIX_REPEAT, allow_heldout=True)
        self.assertEqual(compiled.heldout, "prefix_repeat")

    def test_repeat_above_three_is_rejected(self):
        payload = {
            "schema": {
                "type": "repeat",
                "min": 1,
                "max": 5,
                "child": {"type": "class", "class": "digit", "min": 1, "max": 1},
            }
        }
        with self.assertRaises(SchemaError) as caught:
            compile_schema(payload, allow_heldout=True)
        self.assertEqual(caught.exception.reason, "repeat_too_long")

    def test_confusable_pair_disagrees_on_some_strings(self):
        word = {"type": "literal_slot", "role": "word", "class": "upper", "min": 3, "max": 3}
        digits = {"type": "class", "class": "digit", "min": 3, "max": 3}
        suffix = {"type": "literal_slot", "role": "suffix", "class": "upper", "min": 2, "max": 2}
        left = compile_schema({"schema": {"type": "seq", "children": [word, suffix]}}, allow_heldout=True)
        right = compile_schema(
            {"schema": {"type": "alt", "children": [word, digits]}},
            allow_heldout=True,
        )
        rng = random.Random(3)
        self.assertTrue(languages_differ(Language(left, rng), Language(right, rng), rng))

    def test_atomic_batch_matches_the_frozen_sampler(self):
        left = sample_recipe_batch("d0", random.Random(0), 1, 12, None)
        right = sample_batch(TRAIN_FAMILIES[0], random.Random(0), 12)
        self.assertEqual([episode.query for episode in left], [episode.query for episode in right])
        self.assertEqual([episode.label for episode in left], [episode.label for episode in right])

    def test_rename_leaves_relational_features_unchanged(self):
        compiled = d1_bank()[0]
        episode = sample_schema_episode(compiled, random.Random(4), 1, "class_repeat")
        renamed = rename_episode(episode.as_episode(), random.Random(5), True)
        original = featurize([episode.as_episode()], max_len=32, max_support=4)
        shifted = featurize([renamed], max_len=32, max_support=4)
        for lhs, rhs in zip(original, shifted):
            self.assertTrue(torch.equal(lhs, rhs))


class TeacherParseTest(unittest.TestCase):
    def test_fenced_json_and_budget_guard(self):
        payload = extract_payload('```json\n{"schemas": [{"id": "a"}]}\n```')
        self.assertEqual(payload["schemas"][0]["id"], "a")
        self.assertAlmostEqual(estimate_cost("gpt-5.6-terra", {"input_tokens": 1_000_000, "output_tokens": 1_000_000}), 14.0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "teacher_runs"
            ledger.mkdir()
            (ledger / "ledger.json").write_text(json.dumps({"spent_usd": 11.9, "calls": 1, "budget_usd": 12.0, "events": []}))
            with self.assertRaises(RuntimeError):
                reserve(root, "gpt-5.6-terra", 20_000)


if __name__ == "__main__":
    unittest.main()
