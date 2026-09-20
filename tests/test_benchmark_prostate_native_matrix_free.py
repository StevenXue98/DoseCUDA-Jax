"""GPU-free checks for the bounded native-resolution benchmark controller."""

import unittest
from time import perf_counter

import numpy as np

from tests.benchmark_prostate_native_matrix_free import partial_solve


class _Sampler:
    peak_mib = 0


class PartialSolveTests(unittest.TestCase):
    def test_keeps_initial_and_actual_final_iteration(self):
        class Quadratic:
            @staticmethod
            def value_and_gradient(weights, loss):
                value = float(np.sum((weights - 2.0) ** 2))
                gradient = 2.0 * (weights - 2.0)
                return value, gradient

        initial = np.asarray([0.5, 1.0], dtype=np.float64)
        states, times, result = partial_solve(
            Quadratic(), None, initial, max_iterations=20,
            deadline=perf_counter() + 10, sampler=_Sampler(),
            max_used_gpu_mib=9500)
        np.testing.assert_array_equal(states[0], initial)
        self.assertIn(result["iterations"], states)
        self.assertIn(result["iterations"], times)
        self.assertLess(result["iterations"], 20)
        np.testing.assert_allclose(states[result["iterations"]], [2.0, 2.0],
                                   atol=1e-8)

    def test_deadline_prevents_callback(self):
        class NeverCalled:
            @staticmethod
            def value_and_gradient(weights, loss):
                raise AssertionError("expired callback should not be called")

        with self.assertRaises(TimeoutError):
            partial_solve(NeverCalled(), None, np.asarray([1.0]),
                          max_iterations=5, deadline=perf_counter() - 1,
                          sampler=_Sampler(), max_used_gpu_mib=9500)


if __name__ == "__main__":
    unittest.main()
