"""Small CPU checks for the patient research objective."""

import unittest

import numpy as np

from DoseCUDA.impt_weight_optimization import InfluenceMatrixPlanDose
from tests.run_prostate_nominal_plan import make_loss, solve_scaled_lbfgsb


class ProstateNominalLossTests(unittest.TestCase):
    def test_value_and_adjoint(self):
        high = np.array([True, False, False, False])
        low = np.array([False, True, False, False])
        oar = np.array([False, False, True, True])
        loss = make_loss([
            ("high", high, 1.0, 1.0, False),
            ("low", low, 0.8, 2.0, False),
            ("oar", oar, 0.3, 0.5, True),
        ])
        dose = np.array([0.7, 1.1, 0.1, 0.8])
        value, gradient = loss(dose)
        self.assertAlmostEqual(value, 0.09 + 2 * 0.09 + 0.5 * 0.25 / 2)
        h = 1.0e-6
        for index in range(4):
            direction = np.eye(4)[index]
            difference = (loss(dose + h * direction)[0] -
                          loss(dose - h * direction)[0]) / (2 * h)
            self.assertAlmostEqual(gradient[index], difference, places=8)

    def test_selected_matrix_rows(self):
        class Grid:
            size = (1, 1, 4)

        class Beam:
            dose_grid = Grid()

        class FixedPlan:
            beams = (Beam(),)
            n_spots = 2
            weights = np.array([0.5, 0.25])

            def dose(self, weights):
                columns = np.array([[1.0, 0.5], [0.2, 1.3],
                                    [0.6, 0.7], [0.0, 0.0]])
                return (columns @ weights).reshape(Grid.size)

        selected = np.array([[[True, False, True, False]]])
        matrix = InfluenceMatrixPlanDose(FixedPlan(), max_elements=4,
                                         row_mask=selected)
        np.testing.assert_allclose(matrix.matrix,
                                   [[1.0, 0.5], [0.6, 0.7]], atol=1e-7)
        np.testing.assert_allclose(matrix.dose([0.2, 0.3]), [0.35, 0.33],
                                   atol=1e-7)
        np.testing.assert_allclose(matrix.weight_vjp([2.0, -1.0]),
                                   [1.4, 0.3], atol=1e-7)

    def test_scaled_lbfgsb_nonnegative_quadratic(self):
        def callback(weights):
            error = weights - np.array([0.5, -0.2])
            return float(np.dot(error, error)), 2.0 * error

        result = solve_scaled_lbfgsb(callback, np.array([0.1, 0.1]), 100)
        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.weights, [0.5, 0.0], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
