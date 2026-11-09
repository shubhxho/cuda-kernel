import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("intensity", Path(__file__).parents[1] / "tools/intensity.py")
intensity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(intensity)


class IntensityTest(unittest.TestCase):
    def test_gemv_is_about_one_flop_per_byte(self):
        g = intensity.gemv(8192, 2048, 8192 * 4)
        self.assertEqual(g["flops"], 2 * 8192 * 2048)
        self.assertEqual(g["weight_bytes"], 8192 * 2048 * 2)
        self.assertEqual(g["bytes"], 8192 * 2048 * 2 + 2048 * 2 + 8192 * 4)
        self.assertGreater(g["flops_per_byte"], 0.99)
        self.assertLess(g["flops_per_byte"], 1.01)

    def test_decode_1b_custom_ffn_bytes(self):
        report = intensity.report()
        shapes = {s["name"]: s for s in report["shapes"]}
        s = shapes["decode-1b-shape"]
        self.assertEqual(s["hidden"], 2048)
        self.assertEqual(s["intermediate"], 8192)
        self.assertEqual(s["projection"]["bytes"], 33591296)
        self.assertEqual(s["custom_ffn"]["bytes"], 100749312)
        self.assertEqual(report["result_class"], "analytical-lower-bound-not-a-benchmark")
        self.assertNotEqual(s["intermediate"], shapes["tinyllama-1.1b"]["intermediate"])

    def test_bounds_are_bytes_over_bandwidth(self):
        report = intensity.report()
        s = report["shapes"][0]
        b = s["bandwidth_lower_bounds_us"][1]
        self.assertEqual(b["bandwidth_id"], "partner-desktop-28gbps-derived")
        expected = s["custom_ffn"]["bytes"] / (448 * 10**9) * 1e6
        self.assertAlmostEqual(b["custom_ffn_us_if_100pct_peak"], expected, places=9)
        self.assertIn("not-a-prediction", b["kind"])

    def test_full_7b_8b_models_not_claimed_to_fit(self):
        report = intensity.report()
        for name in ("decode-7b-shape", "decode-8b-shape"):
            working = next(s for s in report["shapes"] if s["name"] == name)["single_layer_working_set_bytes"]
            self.assertLess(working, 8 * 1024**3)


if __name__ == "__main__":
    unittest.main()
