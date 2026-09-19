"""CPU-only checks for the dose-space target/OAR objective."""

import unittest

import numpy as np

from DoseCUDA.impt_weight_optimization import (
    bounded_lbfgsb,
    make_target_oar_loss,
    make_target_oar_normal_tissue_loss,
    projected_gradient_descent,
)


class TargetOARLossTests(unittest.TestCase):
    def test_normal_tissue_overdose_value_and_gradient(self):
        target = np.zeros((2, 2, 2), dtype=bool)
        oar = np.zeros_like(target)
        normal = np.zeros_like(target)
        target[0, 0, 0] = True
        oar[0, 0, 1] = True
        normal[0, 1, 0] = True
        loss = make_target_oar_normal_tissue_loss(
            target, 2.0, oar, 1.0, normal, 2.0
        )
        dose = np.zeros(target.shape, dtype=np.float32)
        dose[0, 0, 0] = 1.0
        dose[0, 0, 1] = 2.0
        dose[0, 1, 0] = 3.0
        value, gradient = loss(dose)
        self.assertAlmostEqual(value, 0.75)
        np.testing.assert_allclose(
            gradient[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
            [-0.5, 0.5, 0.5],
        )
        self.assertEqual(np.count_nonzero(gradient), 3)

        step = np.float32(1.0e-3)
        for index in ((0, 0, 0), (0, 0, 1), (0, 1, 0)):
            plus, minus = dose.copy(), dose.copy()
            plus[index] += step
            minus[index] -= step
            finite_difference = (loss(plus)[0] - loss(minus)[0]) / (2.0 * step)
            self.assertAlmostEqual(
                float(gradient[index]), float(finite_difference), delta=2.0e-4
            )

    def test_normal_tissue_mask_must_not_overlap(self):
        target = np.zeros((2, 2, 2), dtype=bool)
        oar = np.zeros_like(target)
        target[0, 0, 0] = True
        oar[0, 0, 1] = True
        with self.assertRaisesRegex(ValueError, "must exclude target and OAR"):
            make_target_oar_normal_tissue_loss(
                target, 2.0, oar, 1.0, target, 2.0
            )

    def test_value_and_gradient(self):
        target = np.zeros((2, 2, 2), dtype=bool)
        oar = np.zeros_like(target)
        target[0, 0, 0] = True
        oar[0, 1, 0] = True
        loss = make_target_oar_loss(target, 2.0, oar, 1.0, oar_weight=3.0)

        dose = np.zeros(target.shape, dtype=np.float32)
        dose[0, 0, 0] = 1.0
        dose[0, 1, 0] = 2.0
        value, gradient = loss(dose)

        self.assertAlmostEqual(value, 1.0)
        self.assertAlmostEqual(float(gradient[0, 0, 0]), -0.5)
        self.assertAlmostEqual(float(gradient[0, 1, 0]), 1.5)
        self.assertEqual(np.count_nonzero(gradient), 2)

    def test_dose_gradient_matches_central_difference(self):
        target = np.zeros((2, 2, 2), dtype=bool)
        oar = np.zeros_like(target)
        target[0, 0, :] = True
        oar[1, 1, :] = True
        loss = make_target_oar_loss(target, 2.0, oar, 1.0, oar_weight=0.4)
        dose = np.asarray(
            [[[1.2, 2.5], [0.0, 0.0]], [[0.0, 0.0], [1.4, 0.5]]],
            dtype=np.float32,
        )
        _, gradient = loss(dose)
        step = np.float32(1.0e-3)

        for index in ((0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1)):
            plus = dose.copy()
            minus = dose.copy()
            plus[index] += step
            minus[index] -= step
            finite_difference = (loss(plus)[0] - loss(minus)[0]) / (2.0 * step)
            self.assertAlmostEqual(
                float(gradient[index]), float(finite_difference), delta=2.0e-4
            )

    def test_rejects_empty_mask(self):
        mask = np.zeros((2, 2, 2), dtype=bool)
        with self.assertRaisesRegex(ValueError, "target mask must not be empty"):
            make_target_oar_loss(mask, 2.0, np.ones_like(mask), 1.0)

    def test_small_objective_change_does_not_mean_stationarity(self):
        def objective(weights):
            residual = float(weights[0]) - 1.0
            return 0.5 * residual * residual, np.asarray([residual], dtype=np.float32)

        result = projected_gradient_descent(
            objective,
            np.asarray([0.0], dtype=np.float32),
            initial_step=1.0e-7,
            relative_tolerance=1.0e-6,
            gradient_tolerance=1.0e-9,
        )
        self.assertEqual(result.iterations, 1)
        self.assertFalse(result.converged)

    def test_bounded_solver_and_warm_start(self):
        target = np.asarray((1.0, -1.0), dtype=np.float64)

        def objective(weights):
            residual = weights - target
            return 0.5 * float(np.dot(residual, residual)), residual

        result = bounded_lbfgsb(objective, np.asarray((0.0, 2.0)))
        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.weights, (1.0, 0.0), atol=1.0e-6)
        self.assertAlmostEqual(result.objective, 0.5)
        self.assertLess(result.projected_gradient_norm, 1.0e-6)

        warm_start = bounded_lbfgsb(objective, result.weights)
        self.assertTrue(warm_start.converged)
        np.testing.assert_allclose(warm_start.weights, result.weights)

    def test_bounded_solver_rejects_bad_gradient(self):
        with self.assertRaisesRegex(ValueError, "weight gradient"):
            bounded_lbfgsb(lambda weights: (1.0, np.zeros(2)), [1.0])


if __name__ == "__main__":
    unittest.main()
