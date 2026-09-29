import json
import unittest
from pathlib import Path

import torch

from pattern_induction.factory import family_key, load_templates, validate_concept
from prior.expansion import INTEGER_EXPANSION, WIDTH_OPEN, pack_example
from prior.factory_mix import (
    build_split,
    label_candidates,
    load_evidence,
    mix_batch,
    support_sets,
)
from tests.test_factory import CURRENCY, YEAR


ROOT = Path(__file__).resolve().parents[1]

BAY = {
    "domain": "warehouse",
    "concept": "bay code",
    "description": "Two letters from the alphabet, not only the letters that showed up.",
    "semantic_type": "bay",
    "positives": ["AA", "BC"],
    "near_misses": ["A1", "ab"],
    "ambiguous_supports": ["AA", "BC"],
    "program": {"op": "set", "spec": "A-Z", "lo": 2, "hi": 2},
    "neighbors": [{"op": "set", "spec": "ABC", "lo": 2, "hi": 2}],
}

WIDE = {
    "domain": "logs",
    "concept": "request id",
    "description": "An uppercase id whose length is not capped at the samples.",
    "semantic_type": "request",
    "positives": ["AB", "CD"],
    "near_misses": ["A", "ab"],
    "ambiguous_supports": ["AB", "CD"],
    "program": {"op": "cls", "class": "upper", "lo": 2, "hi": 128},
    "neighbors": [{"op": "cls", "class": "upper", "lo": 2, "hi": 2}],
}


def _ambiguous(record: dict, configuration: str):
    stored, reason = validate_concept(record)
    if stored is None:
        raise AssertionError(reason)
    for name, supports in support_sets(stored):
        if name != configuration:
            continue
        labeled = label_candidates(stored, supports)
        if labeled is None:
            continue
        programs, flags = labeled
        packed = pack_example(programs, flags, [True] * len(programs), supports)
        return stored, packed
    raise AssertionError(configuration)


class FactoryMixTests(unittest.TestCase):
    def test_year_window_is_an_integer_expansion_against_the_hull(self):
        _, packed = _ambiguous(YEAR, "ambiguous")
        self.assertTrue(bool(packed["ambiguous"]))
        gold = packed["integer_hyp_gold"] > 0.5
        expand = packed["integer_hyp_expand"] > 0.5
        mask = packed["integer_hyp_mask"] > 0.5
        self.assertTrue(bool((gold & expand & mask).any()))
        self.assertTrue(bool((~gold & ~expand & mask).any()))
        self.assertGreater(packed["integer_hyp"][expand, INTEGER_EXPANSION].abs().sum(), 0)

    def test_open_width_is_the_gold_quantifier(self):
        _, packed = _ambiguous(WIDE, "ambiguous")
        self.assertTrue(bool(packed["ambiguous"]))
        gold = packed["width_hyp_gold"] > 0.5
        mask = packed["width_hyp_mask"] > 0.5
        opened = packed["width_hyp"][:, WIDTH_OPEN] > 0.5
        self.assertTrue(bool((gold & opened & mask).any()))
        self.assertTrue(bool((~gold & ~opened & mask).any()))

    def test_charset_superclass_is_the_gold_alphabet(self):
        _, packed = _ambiguous(BAY, "ambiguous")
        self.assertTrue(bool(packed["ambiguous"]))
        gold = packed["charset_hyp_gold"] > 0.5
        expand = packed["charset_hyp_expand"] > 0.5
        mask = packed["charset_hyp_mask"] > 0.5
        self.assertTrue(bool((gold & expand & mask).any()))
        self.assertTrue(bool((~gold & ~expand & mask).any()))
        self.assertTrue(bool(packed["gold"][0]))
        self.assertFalse(bool(packed["gold"][1]))

    def test_lexicon_neighbor_is_not_probe_perfect(self):
        stored, reason = validate_concept(CURRENCY)
        self.assertIsNotNone(stored, reason)
        found = False
        for _, supports in support_sets(stored):
            labeled = label_candidates(stored, supports)
            if labeled is None:
                continue
            _, flags = labeled
            self.assertTrue(flags[0])
            self.assertIn(False, flags)
            found = True
        self.assertTrue(found)

    def test_builder_keeps_only_the_listed_template(self):
        year, reason = validate_concept(YEAR)
        currency, other = validate_concept(CURRENCY)
        self.assertIsNotNone(year, reason)
        self.assertIsNotNone(currency, other)
        models = load_evidence(ROOT / "results" / "prior" / "runs" / "relative_evidence", [0])
        batch = build_split([year, currency], [year["id"]], models)
        self.assertIsNotNone(batch)
        self.assertIn(year["id"], batch["template_id"])
        self.assertNotIn(currency["id"], batch["template_id"])
        self.assertEqual(tuple(batch["evidence_0"].shape[1:]), (32,))
        self.assertTrue(torch.isfinite(batch["evidence_0"]).all())
        self.assertEqual(batch["evidence_0"].shape[0], len(batch["template_id"]))

    def test_mix_batch_takes_sixteen_from_each_pool(self):
        def fake(count: int, fill: float) -> dict:
            return {
                "ambiguous": torch.ones(count, dtype=torch.bool),
                "mask": torch.ones(count, 32, dtype=torch.bool),
                "gold": torch.zeros(count, 32, dtype=torch.bool),
                "evidence_0": torch.full((count, 32), fill),
            }

        batch = mix_batch(
            fake(4, 1.0),
            fake(3, 2.0),
            seed=0,
            syn_pool=torch.arange(4),
            fac_pool=torch.arange(3),
            batch_size=4,
        )
        self.assertEqual(tuple(batch["evidence"].shape), (4, 32))
        self.assertTrue(torch.all(batch["evidence"][:2] == 1))
        self.assertTrue(torch.all(batch["evidence"][2:] == 2))

    def test_committed_split_keeps_families_apart(self):
        records = load_templates(ROOT)
        split = json.loads((ROOT / "data" / "factory" / "manifests" / "split_v1.json").read_text())
        by_id = {record["id"]: family_key(record) for record in records}
        covered = set(split["train"]) | set(split["validation"]) | set(split["heldout"])
        self.assertEqual(set(by_id), covered)
        groups = {
            name: {by_id[ident] for ident in split[name]} for name in ("train", "validation", "heldout")
        }
        self.assertFalse(groups["train"] & groups["validation"])
        self.assertFalse(groups["train"] & groups["heldout"])
        self.assertFalse(groups["validation"] & groups["heldout"])
        homes = []
        for record in records:
            if family_key(record) != "semver":
                continue
            for name in ("train", "validation", "heldout"):
                if record["id"] in set(split[name]):
                    homes.append(name)
        self.assertEqual(len(set(homes)), 1)


if __name__ == "__main__":
    unittest.main()
