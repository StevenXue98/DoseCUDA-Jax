"""GPU influence-matrix objective checks on an analytic tiny dose operator."""

import unittest

import numpy as np

from DoseCUDA.gpu_influence_matrix import GPUInfluenceMatrixObjective
from DoseCUDA.impt_weight_optimization import (
    bounded_slsqp,
    make_target_oar_normal_tissue_loss,
)


class GPUInfluenceMatrixTests(unittest.TestCase):
    def setUp(self):
        self.shape = (1, 1, 4)
        self.matrix = np.asarray([
            [1.0, 0.5], [0.2, 1.3], [0.6, 0.7], [0.0, 0.0],
        ], dtype=np.float64)
        self.target = np.asarray([[[1, 0, 0, 0]]], dtype=np.uint8)
        self.oar = np.asarray([[[0, 1, 0, 0]]], dtype=np.uint8)
        self.normal = np.asarray([[[0, 0, 1, 0]]], dtype=np.uint8)
        self.objective = GPUInfluenceMatrixObjective(
            self.matrix, self.shape, self.target, self.oar, self.normal,
            prescription=1.0, oar_limit=0.3, normal_limit=0.4,
            oar_weight=2.0, normal_weight=0.5)
        self.cpu_loss = make_target_oar_normal_tissue_loss(
            self.target, 1.0, self.oar, 0.3, self.normal, 0.4,
            oar_weight=2.0, normal_tissue_weight=0.5)

    def test_dose_loss_and_gradient(self):
        for weights in (np.asarray([0.7, 0.2]),
                        np.asarray([0.1, 0.1]),
                        np.asarray([0.5, 0.8])):
            dose = (self.matrix @ weights).reshape(self.shape)
            cpu_value, dose_gradient = self.cpu_loss(dose)
            gpu_value, gpu_gradient = self.objective.value_and_gradient(weights)
            np.testing.assert_allclose(self.objective.dose(weights), dose,
                                       rtol=0, atol=1.0e-14)
            self.assertAlmostEqual(gpu_value, cpu_value, places=13)
            np.testing.assert_allclose(
                gpu_gradient, self.matrix.T @ dose_gradient.ravel(),
                rtol=0, atol=1.0e-13)

    def test_bounded_solver_matches_cpu(self):
        def cpu_callback(weights):
            dose = (self.matrix @ weights).reshape(self.shape)
            value, dose_gradient = self.cpu_loss(dose)
            return value, self.matrix.T @ dose_gradient.ravel()

        settings = dict(weight_scale=1.0, objective_scale=1.0,
                        stationarity_tolerance=1.0e-8)
        initial = np.asarray([0.2, 0.2])
        cpu = bounded_slsqp(cpu_callback, initial, **settings)
        gpu = bounded_slsqp(self.objective.value_and_gradient, initial,
                            **settings)
        self.assertTrue(cpu.converged)
        self.assertTrue(gpu.converged)
        np.testing.assert_allclose(gpu.weights, cpu.weights,
                                   rtol=0, atol=1.0e-9)

    def test_invalid_masks_and_weights(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            GPUInfluenceMatrixObjective(
                self.matrix, self.shape, self.target, self.target,
                self.normal, prescription=1.0, oar_limit=0.3,
                normal_limit=0.4)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            self.objective.value_and_gradient(np.asarray([-0.1, 0.2]))


if __name__ == "__main__":
    unittest.main()
