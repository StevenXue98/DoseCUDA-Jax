"""CPU checks for the partial-inner diagnostic bookkeeping."""

import unittest

import numpy as np

from tests.benchmark_partial_inner_angle import (
    angle_signal, partial_lbfgsb, vector_comparison,
)


class PartialInnerAngleTests(unittest.TestCase):
    def test_partial_snapshots_and_reference_vector(self):
        def quadratic(weights):
            residual = weights - np.array([0.6, 0.2])
            return float(np.dot(residual, residual)), 2.0 * residual

        initial = np.array([0.1, 0.1])
        states, report = partial_lbfgsb(quadratic, initial,
                                        max_seconds=5.0, iterations=5)
        np.testing.assert_array_equal(states[0], initial)
        self.assertGreaterEqual(report["iterations"], 1)
        self.assertLess(quadratic(states[max(states)])[0], quadratic(initial)[0])
        comparison = vector_comparison([1.0, 0.0], [1.0, 0.0])
        self.assertAlmostEqual(comparison["cosine_vs_reference"], 1.0)
        self.assertAlmostEqual(comparison["relative_l2_error"], 0.0)

    def test_central_angle_secant_holds_weights_fixed(self):
        class Probe:
            def __init__(self, angle):
                self.angle = angle

            def dose(self, weights):
                return np.array([self.angle * weights[0]])

        probes = ((Probe(3.0), Probe(1.0)),
                  (Probe(6.0), Probe(4.0)))
        signal = angle_signal(probes, np.array([2.0]),
                              lambda dose: (float(dose[0] ** 2), None), 1.0)
        np.testing.assert_allclose(signal, [16.0, 40.0])


if __name__ == "__main__":
    unittest.main()
