"""GPU-free tests of paired fixed-angle/candidate-angle continuation."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_toy_two_beam_paired import paired_trial  # noqa: E402


class PairedBranchTests(unittest.TestCase):
    def test_equal_step_budget_same_source_and_candidate_can_win(self):
        calls = []

        class FakeExperiment:
            @staticmethod
            def loss(angles, weights):
                return float(weights[0] + (0.0 if angles[0] == 0 else 0.1))

        def solver(experiment, angles, weights, iterations):
            calls.append((float(angles[0]), weights.copy(), iterations))
            weights[0] -= 0.1 if angles[0] == 0 else 0.4
            return weights, {"iterations": iterations}

        source = np.asarray([1.0])
        winner, details = paired_trial(
            FakeExperiment(), np.asarray([0.0, 0.0]), source, 1.0,
            np.asarray([1.0, 0.0]), 10, 1.0, weight_solver=solver)
        self.assertEqual(winner["branch"], "move")
        self.assertAlmostEqual(winner["loss"], 0.7)
        self.assertAlmostEqual(details["stay_loss"], 0.9)
        self.assertEqual(len(calls), 2)
        for _, weights, steps in calls:
            np.testing.assert_array_equal(weights, [1.0])
            self.assertEqual(steps, 10)
        np.testing.assert_array_equal(source, [1.0])

    def test_stay_branch_beats_candidate_after_equal_effort(self):
        class FakeExperiment:
            @staticmethod
            def loss(angles, weights):
                # Candidate beats the *old* 1.0 loss, but not an equally
                # optimized incumbent: this is the earlier confound.
                return float(weights[0] + (0.0 if angles[0] == 0 else 0.1))

        def solver(experiment, angles, weights, iterations):
            weights[0] -= 0.3
            return weights, {"iterations": iterations}

        source = np.asarray([1.0])
        winner, details = paired_trial(
            FakeExperiment(), np.asarray([0.0, 0.0]), source, 1.0,
            np.asarray([1.0, 0.0]), 5, 1.0, weight_solver=solver)
        self.assertEqual(winner["branch"], "stay")
        self.assertAlmostEqual(winner["loss"], 0.7)
        self.assertAlmostEqual(details["trials"][0]["loss"], 0.8)
        self.assertEqual(len(details["trials"]), 6)
        np.testing.assert_array_equal(source, [1.0])

    def test_original_plan_survives_a_nonmonotone_weight_solver(self):
        class FakeExperiment:
            @staticmethod
            def loss(angles, weights):
                return float(weights[0])

        def solver(experiment, angles, weights, iterations):
            weights[0] += 1.0
            return weights, {"iterations": iterations}

        source = np.asarray([1.0])
        winner, _ = paired_trial(
            FakeExperiment(), np.asarray([0.0, 0.0]), source, 1.0,
            np.asarray([1.0, 0.0]), 10, 1.0, weight_solver=solver)
        self.assertEqual(winner["branch"], "original")
        self.assertEqual(winner["loss"], 1.0)
        np.testing.assert_array_equal(winner["weights"], source)


if __name__ == "__main__":
    unittest.main()
