"""GPU-free checks for exact-loss acceptance in alternating toy BAO."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_toy_two_beam_alternating import try_angle_move  # noqa: E402
from compare_toy_two_beam_inner_schedules import shared_probes  # noqa: E402


class AngleAcceptanceTests(unittest.TestCase):
    def test_accepts_only_exact_decrease_with_unchanged_weights(self):
        angles = np.asarray([0.0, 0.0])
        weights = np.asarray([2.0, 3.0], dtype=np.float32)
        seen = []

        def exact_loss(proposal, candidate_weights):
            np.testing.assert_array_equal(candidate_weights, weights)
            seen.append(proposal.copy())
            return float((proposal[0] - 3.0) ** 2 + weights[0])

        moved, trials = try_angle_move(
            exact_loss, angles, weights, 11.0,
            np.asarray([1.0, 0.0]), 12.0)
        self.assertIsNotNone(moved)
        np.testing.assert_array_equal(moved[0], [3.0, 0.0])
        self.assertEqual(len(trials), 3)
        self.assertEqual(len(seen), 3)
        np.testing.assert_array_equal(angles, [0.0, 0.0])
        np.testing.assert_array_equal(weights, [2.0, 3.0])

    def test_rejects_all_trials_if_exact_loss_does_not_improve(self):
        angles = np.asarray([1.0, 2.0])
        weights = np.asarray([0.2], dtype=np.float32)

        def exact_loss(proposal, candidate_weights):
            return 0.5

        moved, trials = try_angle_move(
            exact_loss, angles, weights, 0.5,
            np.asarray([1.0, 0.0]), 8.0)
        self.assertIsNone(moved)
        self.assertEqual(len(trials), 6)
        np.testing.assert_array_equal(angles, [1.0, 2.0])

    def test_change_below_noise_guard_is_rejected(self):
        moved, _ = try_angle_move(
            lambda proposal, weights: 0.5 - 0.5e-7,
            np.asarray([0.0, 0.0]), np.asarray([1.0]), 0.5,
            np.asarray([1.0, 0.0]), 1.0)
        self.assertIsNone(moved)

    def test_same_start_and_cycle_share_probes_across_methods(self):
        first = shared_probes(17, 1, 2, 8)
        np.testing.assert_array_equal(first, shared_probes(17, 1, 2, 8))
        self.assertFalse(np.array_equal(first, shared_probes(17, 1, 3, 8)))
        self.assertFalse(np.array_equal(first, shared_probes(17, 2, 2, 8)))


if __name__ == "__main__":
    unittest.main()
