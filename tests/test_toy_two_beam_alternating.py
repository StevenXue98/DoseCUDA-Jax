"""GPU-free checks for exact-loss acceptance in alternating toy BAO."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_toy_two_beam_alternating import try_angle_move  # noqa: E402
from compare_toy_two_beam_inner_schedules import (  # noqa: E402
    next_current_checkpoint, shared_probes, try_refitted_angle_move,
)


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

    def test_rejected_candidate_refit_cannot_change_incumbent_weights(self):
        class FakeExperiment:
            @staticmethod
            def loss(angles, weights):
                return float(weights[0])

        incumbent = np.asarray([0.5])

        def bad_candidate(experiment, angles, weights, iterations):
            weights[0] = 2.0  # Deliberately mutate the supplied array.
            return weights, {"iterations": iterations}

        moved, trials = try_refitted_angle_move(
            FakeExperiment(), np.asarray([0.0, 0.0]), incumbent, 0.5,
            np.asarray([1.0, 0.0]), 8.0, 20,
            weight_solver=bad_candidate)
        self.assertIsNone(moved)
        self.assertEqual(len(trials), 6)
        np.testing.assert_array_equal(incumbent, [0.5])

    def test_accepted_candidate_retains_its_refitted_weights(self):
        class FakeExperiment:
            @staticmethod
            def loss(angles, weights):
                return float(weights[0])

        incumbent = np.asarray([0.5])

        def good_candidate(experiment, angles, weights, iterations):
            weights[0] = 0.2
            return weights, {"iterations": iterations}

        moved, trials = try_refitted_angle_move(
            FakeExperiment(), np.asarray([0.0, 0.0]), incumbent, 0.5,
            np.asarray([1.0, 0.0]), 8.0, 20,
            weight_solver=good_candidate)
        self.assertIsNotNone(moved)
        np.testing.assert_array_equal(moved[1], [0.2])
        self.assertEqual(moved[2], 0.2)
        self.assertEqual(len(trials), 1)
        np.testing.assert_array_equal(incumbent, [0.5])

    def test_rejected_move_can_extend_incumbent_budget(self):
        self.assertEqual(next_current_checkpoint(5), 10)
        self.assertEqual(next_current_checkpoint(10), 20)
        self.assertEqual(next_current_checkpoint(20), 50)
        self.assertEqual(next_current_checkpoint(50), 100)
        self.assertEqual(next_current_checkpoint(100), 200)
        self.assertIsNone(next_current_checkpoint(200))


if __name__ == "__main__":
    unittest.main()
