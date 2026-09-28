import json
import unittest
from pathlib import Path

from pattern_induction.program import describe
from pattern_induction.select_train import complete_assembly


class SynthesisV1Test(unittest.TestCase):
    def test_recorded_exact_gates(self):
        coverage = json.loads(Path("results/needle/coverage.json").read_text())
        gate = json.loads(Path("results/needle/gate.json").read_text())
        self.assertEqual(coverage["exact_empty"], 0)
        self.assertGreaterEqual(round(coverage["exact_synthesis_oracle"], 3), 0.922)
        self.assertEqual(round(coverage["recoverable_oracle"], 3), 0.922)
        self.assertEqual(coverage["exact_synthesis_oracle"], coverage["recoverable_oracle"])
        self.assertEqual(coverage["remaining_visible"], 0)
        self.assertEqual(coverage["previously_perfect_kept"], 113)
        self.assertEqual(coverage["previously_perfect_lost"], [])
        self.assertEqual(coverage["selector_timeouts"], 0)
        self.assertGreaterEqual(round(coverage["selector_accuracy_given_perfect"], 3), 0.974)
        self.assertEqual(gate["min_invariance"], 1.0)
        self.assertEqual(gate["parameters"], 1226128)

    def test_v1_assembly_does_not_require_a_character_set(self):
        supports = ["INV-1588", "INV-1399", "INV-8698", "INV-7973"]
        built = complete_assembly(supports, [])
        descriptions = [describe(prog) for prog, _score in built.programs]
        self.assertTrue(any("digit{4,4}" in item for item in descriptions))
        self.assertFalse(any(item.startswith("SET") or "SET[" in item or "INT[" in item for item in descriptions))


if __name__ == "__main__":
    unittest.main()
