"""Validate GPU evidence and render a conservative comparison. Stdlib only."""
import argparse
import json
import math
import statistics
from pathlib import Path


def positive(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def validate(report):
    if report.get("schema_version") != 1 or report.get("timing") != "cuda-graph-events":
        raise ValueError("unsupported benchmark schema or timing protocol")
    if report.get("data") not in ("synthetic-fp16-model-shapes", "captured-model-fixture"):
        raise ValueError("unknown data provenance")
    if not report.get("gpu") or not report.get("cases"):
        raise ValueError("missing GPU identity or workload cases")
    for key in ("driver_version", "runtime_version", "cublas_version", "repeats", "samples"):
        if not positive(report.get(key)) or not isinstance(report[key], int):
            raise ValueError(f"missing or invalid {key}")
    if report["samples"] < 3:
        raise ValueError("insufficient holdout samples")
    seen = set()
    for case in report["cases"]:
        identity = (case.get("shape"), case.get("operation"))
        if identity in seen:
            raise ValueError("duplicate benchmark case")
        seen.add(identity)
        if case.get("operation") not in ("projection", "swiglu-ffn"):
            raise ValueError("unknown benchmark operation")
        for key in ("hidden", "intermediate"):
            if not positive(case.get(key)) or not isinstance(case[key], int):
                raise ValueError("invalid workload dimensions")
        for side in ("custom", "baseline"):
            c = case[side]
            values = c["samples_us"]
            if c.get("correct") is not True or len(values) != report["samples"]:
                raise ValueError("incorrect or incomplete candidate")
            if not all(positive(v) for v in values):
                raise ValueError("invalid GPU event timings")
            for key in ("max_abs_error", "max_scaled_error"):
                v = c.get(key)
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                    raise ValueError("invalid correctness evidence")
            if c["max_scaled_error"] > 1:
                raise ValueError("numeric tolerance failed")
            if not math.isclose(c["median_us"], statistics.median(values), rel_tol=1e-7):
                raise ValueError("median does not match raw samples")
        ratio = case["baseline"]["median_us"] / case["custom"]["median_us"]
        if not math.isclose(ratio, case["speedup"], rel_tol=1e-7):
            raise ValueError("speedup does not match timings")
    return report


def render(report):
    validate(report)
    lines = [f"GPU: {report['gpu']} (SM {report['compute_capability']}).",
             f"CUDA runtime {report['runtime_version']}; cuBLAS {report['cublas_version']}.", "",
             f"Data: `{report['data']}`. Resident weights; warmed CUDA Graphs; GPU event timing.",
             "Fresh holdout samples follow tuning. Transfers, CPU reference, and tuning are excluded.", "",
             "| Workload | Operation | Custom µs | cuBLAS µs | Ratio | Evidence |",
             "|---|---|---:|---:|---:|---|"]
    for c in report["cases"]:
        a, b = c["custom"], c["baseline"]
        separated = max(a["samples_us"]) < min(b["samples_us"])
        evidence = "separated win" if separated and c["speedup"] >= 1.05 else (
            "slower" if c["speedup"] < 1 else "inconclusive")
        lines.append(f"| {c['shape']} | {c['operation']} | {a['median_us']:.3f} | "
                     f"{b['median_us']:.3f} | {c['speedup']:.3f}× | {evidence} |")
    lines += ["", "Tail latency (nearest-rank p95 of replay-batch averages):", "",
              "| Workload | Operation | Custom p95 µs | Baseline p95 µs |", "|---|---|---:|---:|"]
    for c in report["cases"]:
        def p95(side):
            xs = sorted(c[side]["samples_us"])
            return xs[math.ceil(0.95 * len(xs)) - 1]
        lines.append(f"| {c['shape']} | {c['operation']} | {p95('custom'):.3f} | {p95('baseline'):.3f} |")
    lines += ["", "A separated win requires at least 5% median speedup and every custom sample below every baseline sample.",
              "This compares the selected cuBLAS GemmEx algorithms. It does not establish a win against cuBLASLt, cuDNN,",
              "TensorRT-LLM, a complete model, or other GPUs. Synthetic shapes are not pretrained-model evidence."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-win", action="store_true")
    args = parser.parse_args()
    report = validate(json.loads(args.input.read_text()))
    text = render(report)
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
    if args.require_win and not any(
        c["speedup"] >= 1.05 and max(c["custom"]["samples_us"]) < min(c["baseline"]["samples_us"])
        for c in report["cases"]
    ):
        raise SystemExit("No separated win measured")


if __name__ == "__main__":
    main()
