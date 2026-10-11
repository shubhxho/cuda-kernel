import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("model_report", Path(__file__).parents[1] / "tools/model_report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def fixture():
    def side(times, calls):
        return {"samples": [{"wall_ms_per_token": t, "gpu_ms_per_token": t/2, "calls": calls} for t in times],
                "wall_median_ms": times[1], "gpu_median_ms": times[1]/2}
    return {"kind": "pretrained-model-decode", "resolved_revision": "test-only-revision", "model": "fixture",
            "gpu": "test-only", "patched_layers": 2, "cases": [{"prompt": "test|prompt", "tokens": 4,
            "exact_greedy_match": True, "logits_close": True, "max_logit_abs_error": .01, "max_logit_scaled_error": .1,
            "baseline": side([9,10,11], 0), "custom": side([4,5,6],8), "wall_speedup": 2}]}


class ModelReportTest(unittest.TestCase):
    def test_valid(self):
        self.assertIn("| separated win |", report.render(fixture()))

    def test_incomplete_route_rejected(self):
        data = fixture()
        data["cases"][0]["custom"]["samples"][0]["calls"] = 1
        with self.assertRaises(ValueError):
            report.render(data)

    def test_wrong_tokens_rejected(self):
        data = fixture()
        data["cases"][0]["exact_greedy_match"] = False
        with self.assertRaises(ValueError):
            report.render(data)

    def test_bad_logits_rejected(self):
        data = fixture()
        data["cases"][0]["max_logit_scaled_error"] = 1.01
        with self.assertRaises(ValueError):
            report.render(data)

    def test_fake_latency_rejected(self):
        data = fixture()
        data["cases"][0]["wall_speedup"] = 10
        with self.assertRaises(ValueError):
            report.render(data)
