import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEAL_SHA = "f370a1707179ede5212d360751c5fb23ae7b6f507ec75665cd727dffc920ba9d"


class BenchESealTest(unittest.TestCase):
    def test_the_split_was_scored_once_and_the_file_is_unchanged(self):
        raw = (ROOT / "data" / "bench_e.jsonl").read_bytes()
        seal = json.loads((ROOT / "results" / "needle" / "bench_e_seal.json").read_text())
        self.assertEqual(hashlib.sha256(raw).hexdigest(), SEAL_SHA)
        self.assertEqual(seal["sha256"], SEAL_SHA)
        self.assertEqual(sum(1 for line in raw.splitlines() if line.strip()), 100)
        self.assertTrue(seal["scored"])
        self.assertEqual(seal["n"], 100)
        self.assertAlmostEqual(seal["accuracy"], 0.9847916666666667)
        self.assertIn("No tuning afterward", seal["note"])

    def test_a_scored_seal_is_not_rewritten(self):
        from pattern_induction.bench_e import seal_bench_e

        with self.assertRaises(RuntimeError):
            seal_bench_e(ROOT)


if __name__ == "__main__":
    unittest.main()
