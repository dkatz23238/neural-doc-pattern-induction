import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BenchESealTest(unittest.TestCase):
    def test_the_split_is_sealed_and_unscored(self):
        raw = (ROOT / "data" / "bench_e.jsonl").read_bytes()
        seal = json.loads((ROOT / "results" / "needle" / "bench_e_seal.json").read_text())
        self.assertEqual(hashlib.sha256(raw).hexdigest(), seal["sha256"])
        self.assertEqual(sum(1 for line in raw.splitlines() if line.strip()), seal["n"])
        self.assertFalse(seal["scored"])
        self.assertFalse(seal["inspected"])
        self.assertGreater(seal["n"], 0)


if __name__ == "__main__":
    unittest.main()
