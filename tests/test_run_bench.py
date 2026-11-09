import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("run_bench", Path(__file__).parents[1] / "tools/run_bench.py")
run_bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_bench)


class ProbeTest(unittest.TestCase):
    def test_probe_records_missing_gpu(self):
        env = run_bench.probe()
        self.assertEqual(env["result_class"], "environment-probe")
        self.assertIn("gpu_present", env)
        self.assertIn("cuda_compiler_present", env)
        self.assertIn("nvidia_smi", env)
        if not env["gpu_present"]:
            self.assertFalse(env["nvidia_smi"].get("available", True) and env.get("torch_cuda", {}).get("is_available"))

    def test_probe_only_writes_environment_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Direct call of probe writer path.
            path = Path(tmp) / "environment.json"
            env = run_bench.probe()
            run_bench.write_json(path, env)
            text = path.read_text()
            self.assertIn("environment-probe", text)
            self.assertNotIn("RTX 5060", text)


if __name__ == "__main__":
    unittest.main()
