import unittest

from phase_transition.statistics.fixed_response import (
    exact_sign_flip_test,
    percentile_bootstrap_ci,
)


class StatisticsTests(unittest.TestCase):
    def test_exact_sign_flip_enumerates_all_assignments(self):
        result = exact_sign_flip_test({"q1": 1.0, "q2": 2.0, "q3": 3.0}, 1)
        self.assertEqual(result["n_combinations"], 8)
        self.assertAlmostEqual(result["observed_mean"], 2.0)
        self.assertAlmostEqual(result["p_one_sided_primary"], 0.125)

    def test_bootstrap_is_deterministic(self):
        values = {"q1": -1.0, "q2": -2.0, "q3": -3.0}
        first = percentile_bootstrap_ci(values, 100, 42, "effect")
        second = percentile_bootstrap_ci(values, 100, 42, "effect")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
