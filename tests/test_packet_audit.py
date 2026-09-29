import unittest

import numpy as np

from prior.packet_audit import (
    _fingerprint,
    group_labels,
    packet_stats,
    pool_ceiling,
    row_keys,
)


class PacketAuditTest(unittest.TestCase):
    def test_exact_rows_share_a_key_and_negative_zero_does_not_split_them(self):
        left = np.array([[0.0, 1.0], [-0.0, 1.0]], dtype=np.float32)
        right = np.array([[1.0, 1.0]], dtype=np.float32)
        keys = row_keys(left)
        self.assertEqual(keys[0], keys[1])
        self.assertNotEqual(keys[0], row_keys(right)[0])

    def test_a_conflicting_packet_has_entropy_and_a_majority_tie(self):
        values = np.array([[1.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        gold = np.array([True, False, True])
        _keys, positive, negative = group_labels(row_keys(values), gold)
        stats = packet_stats(positive, negative)
        self.assertEqual(stats["unique"], 2)
        self.assertEqual(stats["conflict_packets"], 1)
        self.assertAlmostEqual(stats["entropy"], 1.0 * 2 / 3, places=6)
        self.assertAlmostEqual(stats["majority_accuracy"], 2 / 3, places=6)

    def test_pool_ceiling_is_exact_when_the_gold_packet_is_unique(self):
        scores = np.array([[0.9, 0.1], [0.4, 0.4]])
        gold = np.array([[True, False], [True, False]])
        alive = np.ones_like(gold)
        ceiling = pool_ceiling(scores, gold, alive)
        self.assertAlmostEqual(ceiling["top1"], 0.75, places=6)
        self.assertAlmostEqual(ceiling["strict"], 0.5, places=6)

    def test_a_repeated_node_does_not_cancel_in_the_program_signature(self):
        single = _fingerprint(np.array([[[1, 0]]]), np.array([[[True, False]]]))
        double = _fingerprint(np.array([[[1, 1]]]), np.array([[[True, True]]]))
        swapped = _fingerprint(np.array([[[2, 1]]]), np.array([[[True, True]]]))
        ordered = _fingerprint(np.array([[[1, 2]]]), np.array([[[True, True]]]))
        self.assertNotEqual(bytes(single.reshape(-1)[0]), bytes(double.reshape(-1)[0]))
        self.assertEqual(bytes(swapped.reshape(-1)[0]), bytes(ordered.reshape(-1)[0]))


if __name__ == "__main__":
    unittest.main()
