import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("report", Path(__file__).parents[1] / "tools/report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def fixture():
    def side(name, values):
        return {"name": name, "correct": True, "samples_us": values, "median_us": values[1],
                "max_abs_error": 0.0001, "max_scaled_error": 0.01}
    return {"schema_version": 1, "timing": "cuda-graph-events", "gpu": "test-only",
            "compute_capability": "8.0", "data": "synthetic-fp16-model-shapes",
            "driver_version": 12080, "runtime_version": 12080, "cublas_version": 120800,
            "repeats": 100, "samples": 3, "cases": [{"shape": "fixture", "operation": "projection", "hidden": 64, "intermediate": 128,
            "custom": side("custom", [4, 5, 6]), "baseline": side("cublas", [9, 10, 11]), "speedup": 2}]}


class ReportTest(unittest.TestCase):
    def test_boolean_error(self):
        data=fixture();data["cases"][0]["custom"]["max_scaled_error"]=True
        with self.assertRaises(ValueError): report.validate(data)

    def test_duplicate_cases(self):
        data=fixture();data["cases"].append(copy.deepcopy(data["cases"][0]))
        with self.assertRaises(ValueError): report.validate(data)

    def test_valid(self):
        self.assertIn("separated win", report.render(fixture()))

    def test_reject_false_claims(self):
        for key, value in [("correct", False), ("median_us", 99), ("max_scaled_error", 1.1),
                           ("max_abs_error", float("nan")), ("samples_us", [1, -1, 2])]:
            bad = fixture()
            bad["cases"][0]["custom"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                report.validate(bad)

    def test_reject_invented_speedup(self):
        bad = fixture()
        bad["cases"][0]["speedup"] = 100
        with self.assertRaises(ValueError):
            report.validate(bad)

    def test_noise_not_win(self):
        data = fixture()
        data["cases"][0]["custom"]["samples_us"] = [4, 5, 12]
        self.assertNotIn("| separated win |", report.render(data))


if __name__ == "__main__":
    unittest.main()
