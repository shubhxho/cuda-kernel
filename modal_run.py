"""One bounded GPU job; no deployment, persistent volume, or background worker."""
from pathlib import Path
import json
import modal

ROOT = Path(__file__).resolve().parent
image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("build-essential", "cmake", "ninja-build")
    .pip_install("torch==2.7.1", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("transformers==4.51.3", "numpy==2.2.6", "safetensors==0.5.3", "ninja==1.11.1.4", "cmake==3.31.6")
    .env({"TORCH_CUDA_ARCH_LIST": "8.9", "MAX_JOBS": "2"})
    .add_local_file(ROOT / "CMakeLists.txt", "/workspace/CMakeLists.txt", copy=True)
    .add_local_dir(ROOT / "include", "/workspace/include", copy=True)
    .add_local_dir(ROOT / "src", "/workspace/src", copy=True)
    .add_local_dir(ROOT / "tools", "/workspace/tools", copy=True, ignore=["**/__pycache__/**"])
    .add_local_dir(ROOT / "tests", "/workspace/tests", copy=True, ignore=["**/__pycache__/**"])
    .run_commands("cmake -S /workspace -B /workspace/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89",
                  "cmake --build /workspace/build -j2")
)
app = modal.App("cuda-decode-bench", image=image)


@app.function(gpu="L4", cpu=4, memory=8192, timeout=900, retries=0, max_containers=1, scaledown_window=2)
def benchmark(with_model: bool = True):
    import hashlib
    import subprocess
    directory = Path("/workspace/results")
    directory.mkdir(exist_ok=True)
    commands = [
        ["nvidia-smi"],
        ["ctest", "--test-dir", "/workspace/build", "--output-on-failure"],
        ["python", "-m", "unittest", "discover", "-s", "/workspace/tests", "-v"],
        ["/workspace/build/decode_bench", "--suite", "correctness", "--seed", "101", "--repeats", "5", "--samples", "3"],
        ["/workspace/build/decode_bench", "--suite", "decode", "--output", str(directory / "kernels.json")],
        ["python", "/workspace/tools/report.py", str(directory / "kernels.json"), "--output", str(directory / "kernels.md")],
    ]
    if with_model:
        commands.append(["python", "/workspace/tools/gpu_validate.py"])
        commands.append(["python", "/workspace/tools/model_bench.py", "--output", str(directory / "model.json")])
        commands.append(["python", "/workspace/tools/model_report.py", str(directory / "model.json"), "--output", str(directory / "model.md")])
    logs = []
    for command in commands:
        process = subprocess.run(command, cwd="/workspace", text=True, capture_output=True, timeout=600)
        logs.append({"command": command, "exit_code": process.returncode, "stdout": process.stdout, "stderr": process.stderr})
        if process.returncode:
            return {"ok": False, "logs": logs,
                    "files": {p.name: p.read_text() for p in directory.iterdir() if p.is_file()}}
    source_paths = [p for folder in ("src", "include", "tools") for p in Path("/workspace", folder).rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts]
    provenance = {str(p.relative_to("/workspace")): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return {"ok": True, "logs": logs, "files": {p.name: p.read_text() for p in directory.iterdir() if p.is_file()}}


@app.local_entrypoint()
def main(kernels_only: bool = False):
    result = benchmark.remote(with_model=not kernels_only)
    directory = ROOT / "results"
    directory.mkdir(exist_ok=True)
    (directory / "run-log.json").write_text(json.dumps(result["logs"], indent=2) + "\n")
    for name, data in result["files"].items():
        (directory / name).write_text(data)
    if not result["ok"]:
        raise RuntimeError("GPU validation failed; see results/run-log.json")
    print(f"Validated GPU evidence saved in {directory}")
