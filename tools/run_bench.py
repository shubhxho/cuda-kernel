"""Reproducible runner: environment probe, then GPU gates only if a device exists.

Does not download models unless --model is passed. Does not call paid cloud APIs.
Writes JSON that labels measured runs separately from analytical bounds.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd, timeout=60, check=False):
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, cwd=ROOT)
    except FileNotFoundError as exc:
        return {"command": cmd, "error": f"missing executable: {exc}", "exit_code": 127}
    except subprocess.TimeoutExpired:
        return {"command": cmd, "error": "timeout", "exit_code": 124}
    out = {"command": cmd, "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    if check and proc.returncode:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}")
    return out


def which(name):
    path = shutil.which(name)
    return path


def nvidia_smi_query():
    if not which("nvidia-smi"):
        return {"available": False, "reason": "nvidia-smi not found"}
    query = (
        "name,uuid,driver_version,memory.total,memory.free,compute_cap,"
        "clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,"
        "power.draw,power.limit,persistence_mode,pstate,clocks_throttle_reasons.active"
    )
    raw = run(["nvidia-smi", f"--query-gpu={query}", "--format=csv"], timeout=30)
    listing = run(["nvidia-smi"], timeout=30)
    return {"available": raw.get("exit_code") == 0, "query": raw, "listing": listing}


def probe():
    nvcc = which("nvcc")
    nvcc_version = run(["nvcc", "--version"]) if nvcc else {"available": False}
    cmake = which("cmake")
    gxx = which("g++")
    info = {
        "schema_version": 1,
        "result_class": "environment-probe",
        "retrieved": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "cwd": str(ROOT),
        "executables": {
            "nvcc": nvcc,
            "cmake": cmake,
            "g++": gxx,
            "nvidia-smi": which("nvidia-smi"),
            "decode_bench": str(ROOT / "build/decode_bench") if (ROOT / "build/decode_bench").exists() else None,
        },
        "nvcc_version": nvcc_version,
        "cmake_version": run(["cmake", "--version"]) if cmake else {"available": False},
        "gxx_version": run(["g++", "--version"]) if gxx else {"available": False},
        "nvidia_smi": nvidia_smi_query(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda": None,
    }
    try:
        import torch
        info["torch_cuda"] = {
            "torch": torch.__version__,
            "cuda_compiled": getattr(torch.version, "cuda", None),
            "is_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        info["torch_cuda"] = {"available": False, "reason": str(exc)}
    info["gpu_present"] = bool(
        info["nvidia_smi"].get("available")
        or (info["torch_cuda"] and info["torch_cuda"].get("is_available"))
    )
    info["cuda_compiler_present"] = bool(nvcc)
    return info


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=ROOT / "results")
    p.add_argument("--probe-only", action="store_true")
    p.add_argument("--model", action="store_true")
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--arch", default=os.environ.get("CMAKE_CUDA_ARCHITECTURES", ""))
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    env = probe()
    write_json(a.output_dir / "environment.json", env)
    print(f"GPU present: {env['gpu_present']}; nvcc: {env['cuda_compiler_present']}")
    if a.probe_only or not env["gpu_present"] or not env["cuda_compiler_present"]:
        if not env["gpu_present"] or not env["cuda_compiler_present"]:
            print("Skipping CUDA execution: no NVIDIA GPU and/or no nvcc in this environment.")
            print("Analytical bounds: python3 tools/intensity.py")
        return 0
    if not a.skip_build:
        cmake = ["cmake", "-S", str(ROOT), "-B", str(ROOT / "build"), "-DCMAKE_BUILD_TYPE=Release"]
        if a.arch:
            cmake.append(f"-DCMAKE_CUDA_ARCHITECTURES={a.arch}")
        run(cmake, timeout=180, check=True)
        run(["cmake", "--build", str(ROOT / "build"), "-j2"], timeout=600, check=True)
    binary = ROOT / "build/decode_bench"
    if not binary.exists():
        raise SystemExit("decode_bench missing after build")
    run(["ctest", "--test-dir", str(ROOT / "build"), "--output-on-failure"], timeout=300, check=True)
    for suite in ("correctness", "smoke", "tinyllama", "decode"):
        out = a.output_dir / f"kernels-{suite}.json"
        cmd = [str(binary), "--suite", suite, "--include-transfers", "--output", str(out)]
        if suite == "correctness":
            cmd += ["--repeats", "5", "--samples", "3"]
        run(cmd, timeout=900, check=True)
        md = out.with_suffix(".md")
        run([sys.executable, str(ROOT / "tools/report.py"), str(out), "--output", str(md)], timeout=60, check=True)
        print(f"wrote {out}")
    if a.model:
        run([sys.executable, str(ROOT / "tools/gpu_validate.py")], timeout=600, check=True)
        model_json = a.output_dir / "model.json"
        run([sys.executable, str(ROOT / "tools/model_bench.py"), "--output", str(model_json)], timeout=900, check=True)
        run([sys.executable, str(ROOT / "tools/model_report.py"), str(model_json),
             "--output", str(model_json.with_suffix(".md"))], timeout=60, check=True)
    print(f"Measured GPU evidence saved in {a.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
