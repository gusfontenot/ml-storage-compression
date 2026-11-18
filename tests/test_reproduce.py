"""Result checks."""
import copy
import unittest

from examples.reproduce_results import compare


class ReproduceTests(unittest.TestCase):
    def test_allows_timing_changes(self):
        expected = {"ml": {"mean_prediction_us": 5, "savings": .25, "counts": {"zstd": 80}}}
        actual = copy.deepcopy(expected)
        actual["ml"]["mean_prediction_us"] = 100
        compare(actual, expected, "metrics")

    def test_rejects_result_changes(self):
        expected = {"ml": {"savings": .25, "counts": {"zstd": 80}}}
        for change in ("savings", "counts"):
            actual = copy.deepcopy(expected)
            if change == "savings":
                actual["ml"]["savings"] = .5
            else:
                actual["ml"]["counts"]["zstd"] = 79
            with self.assertRaisesRegex(ValueError, "Result mismatch"):
                compare(actual, expected, "metrics")


if __name__ == "__main__":
    unittest.main()
