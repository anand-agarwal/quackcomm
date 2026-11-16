import math
import unittest

from pp_core.tof import RangeFilter, assess_range


class TofTests(unittest.TestCase):
    def test_close_reading_is_dangerous(self):
        status = assess_range(0.12, 0.03, 1.2, 0.22, 0.32, 0.01, 0.30)
        self.assertTrue(status["valid"])
        self.assertTrue(status["dangerously_close"])
        self.assertFalse(status["clear"])

    def test_far_reading_is_clear(self):
        status = assess_range(0.80, 0.03, 1.2, 0.22, 0.32, 0.01, 0.30)
        self.assertTrue(status["clear"])
        self.assertFalse(status["dangerously_close"])

    def test_nan_is_invalid(self):
        status = assess_range(float("nan"), 0.03, 1.2, 0.22, 0.32, 0.01, 0.30)
        self.assertFalse(status["valid"])
        self.assertFalse(status["clear"])
        self.assertEqual(status["reason"], "nan")

    def test_negative_is_invalid(self):
        status = assess_range(-0.1, 0.03, 1.2, 0.22, 0.32, 0.01, 0.30)
        self.assertFalse(status["valid"])

    def test_below_minimum_is_treated_as_close(self):
        status = assess_range(0.01, 0.03, 1.2, 0.22, 0.32, 0.01, 0.30)
        self.assertTrue(status["dangerously_close"])

    def test_positive_infinity_means_no_return(self):
        status = assess_range(float("inf"), 0.03, 1.2, 0.22, 0.32, 0.01, 0.30)
        self.assertTrue(status["valid"])
        self.assertTrue(status["clear"])
        self.assertFalse(status["dangerously_close"])

    def test_stale_clear_reading_is_not_clear(self):
        status = assess_range(0.90, 0.03, 1.2, 0.22, 0.32, 1.0, 0.30)
        self.assertTrue(status["stale"])
        self.assertFalse(status["clear"])

    def test_median_filter_rejects_a_single_spike(self):
        filtered = RangeFilter(3)
        filtered.push(0.80)
        filtered.push(0.05)
        value = filtered.push(0.82)
        self.assertTrue(math.isclose(value, 0.80))


if __name__ == "__main__":
    unittest.main()
