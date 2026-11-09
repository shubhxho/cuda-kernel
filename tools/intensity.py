"""Arithmetic intensity and analytical DRAM lower bounds for the decode kernels.

This is not a benchmark. It never claims RTX 5060 latency. Peak-bandwidth
divisions are lower bounds under explicit assumptions, labeled as such.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FP16 = 2
FP32 = 4
# SiLU + multiply is tiny next to the matvec; counted so the bound is not
# pretending the epilogue is free. Four FLOPs is an order-of-magnitude stand-in.
SILU_FLOPS = 4
GATE_MUL_FLOPS = 1

SHAPES = (
    {"name": "decode-1b-shape", "hidden": 2048, "intermediate": 8192,
     "note": "Benchmark suite shape. Not TinyLlama; TinyLlama intermediate is 5632."},
    {"name": "tinyllama-1.1b", "hidden": 2048, "intermediate": 5632,
     "note": "TinyLlama-1.1B-Chat-v1.0 MLP dims used by tools/model_bench.py."},
    {"name": "decode-7b-shape", "hidden": 4096, "intermediate": 11008,
     "note": "Llama-2-7B-class FFN shape. One layer, not the full 7B model."},
    {"name": "decode-8b-shape", "hidden": 4096, "intermediate": 14336,
     "note": "Llama-3-8B-class FFN shape. One layer, not the full 8B model."},
    {"name": "smoke", "hidden": 512, "intermediate": 1024,
     "note": "Smoke shape. Launch overhead is a larger fraction of runtime."},
)

# Official NVIDIA laptop spec table lists 384 GB/s. Desktop family spec table
# retrieved 2026-11-09 lists 128-bit GDDR7 but not GB/s. Partner 28 Gbps on
# that 128-bit bus derives 448 GB/s. See docs/rtx-5060.md.
BANDWIDTHS = (
    {"id": "nvidia-laptop-5060-published",
     "bytes_per_sec": 384 * 10**9,
     "label": "384 GB/s",
     "kind": "nvidia-published-laptop",
     "source": "https://www.nvidia.com/en-us/geforce/laptops/50-series/"},
    {"id": "partner-desktop-28gbps-derived",
     "bytes_per_sec": 448 * 10**9,
     "label": "448 GB/s",
     "kind": "derived-from-partner-rate-and-nvidia-bus",
     "source": "ASUS TUF RTX 5060 28 Gbps * NVIDIA 128-bit interface / 8"},
)


def gemv(rows: int, cols: int, out_bytes: int) -> dict:
    flops = 2 * rows * cols
    weight_bytes = rows * cols * FP16
    input_bytes = cols * FP16
    bytes_ = weight_bytes + input_bytes + out_bytes
    return {
        "flops": flops,
        "weight_bytes": weight_bytes,
        "input_bytes": input_bytes,
        "output_bytes": out_bytes,
        "bytes": bytes_,
        "flops_per_byte": flops / bytes_,
    }


def fused_gate(hidden: int, intermediate: int) -> dict:
    # Two FP16 matvecs sharing x, SiLU(gate)*up, FP16 hidden write, two biases.
    base = gemv(intermediate, hidden, intermediate * FP16)
    extra_weights = intermediate * hidden * FP16
    epilogue_flops = intermediate * (SILU_FLOPS + GATE_MUL_FLOPS)
    bias_bytes = 2 * intermediate * FP16
    flops = base["flops"] * 2 + epilogue_flops
    bytes_ = base["bytes"] + extra_weights + bias_bytes
    return {
        "flops": flops,
        "weight_bytes": base["weight_bytes"] + extra_weights,
        "input_bytes": base["input_bytes"],
        "output_bytes": base["output_bytes"],
        "bias_bytes": bias_bytes,
        "epilogue_flops": epilogue_flops,
        "bytes": bytes_,
        "flops_per_byte": flops / bytes_,
    }


def down_residual(hidden: int, intermediate: int) -> dict:
    base = gemv(hidden, intermediate, hidden * FP32)
    residual_bytes = hidden * FP16
    bias_bytes = hidden * FP16
    bytes_ = base["bytes"] + residual_bytes + bias_bytes
    return {
        "flops": base["flops"],
        "weight_bytes": base["weight_bytes"],
        "input_bytes": base["input_bytes"],
        "output_bytes": base["output_bytes"],
        "residual_bytes": residual_bytes,
        "bias_bytes": bias_bytes,
        "bytes": bytes_,
        "flops_per_byte": base["flops"] / bytes_,
    }


def cublas_ffn_extra_bytes(hidden: int, intermediate: int) -> int:
    # Two FP32 GEMV outputs written then reread by the gate epilogue, plus
    # residual epilogue rereading y. Order-of-magnitude library extra traffic.
    return 2 * intermediate * FP32 * 2 + hidden * FP32


def us_at_bandwidth(byte_count: int, bytes_per_sec: float) -> float:
    return byte_count / bytes_per_sec * 1e6


def analyze_shape(shape: dict) -> dict:
    h, i = shape["hidden"], shape["intermediate"]
    projection = gemv(i, h, i * FP32)
    gate = fused_gate(h, i)
    down = down_residual(h, i)
    custom_ffn = {
        "flops": gate["flops"] + down["flops"],
        "bytes": gate["bytes"] + down["bytes"],
        "weight_bytes": gate["weight_bytes"] + down["weight_bytes"],
    }
    custom_ffn["flops_per_byte"] = custom_ffn["flops"] / custom_ffn["bytes"]
    library_ffn_bytes = custom_ffn["bytes"] + cublas_ffn_extra_bytes(h, i)
    working_set = (
        3 * h * i * FP16  # gate, up, down weights
        + h * FP16 + i * FP16  # x and hidden
        + 2 * i * FP32 + h * FP32  # library-sized scratch
        + (i + h) * FP16  # biases
    )
    bounds = []
    for bw in BANDWIDTHS:
        bounds.append({
            "bandwidth_id": bw["id"],
            "bandwidth_label": bw["label"],
            "kind": "analytical-lower-bound-not-a-prediction",
            "projection_us_if_100pct_peak": us_at_bandwidth(projection["bytes"], bw["bytes_per_sec"]),
            "custom_ffn_us_if_100pct_peak": us_at_bandwidth(custom_ffn["bytes"], bw["bytes_per_sec"]),
            "assumption": "100% of named peak bandwidth, one streaming pass, no launch cost",
        })
    return {
        **shape,
        "projection": projection,
        "fused_gate": gate,
        "down_residual": down,
        "custom_ffn": custom_ffn,
        "library_extra_bytes_vs_fused": cublas_ffn_extra_bytes(h, i),
        "library_ffn_bytes_estimate": library_ffn_bytes,
        "single_layer_working_set_bytes": working_set,
        "bandwidth_lower_bounds_us": bounds,
    }


def report() -> dict:
    return {
        "schema_version": 1,
        "result_class": "analytical-lower-bound-not-a-benchmark",
        "label": "ESTIMATES and analytical bounds only. Not RTX 5060 measurements.",
        "retrieved": "2026-11-09",
        "precision": {
            "weights": "fp16",
            "input": "fp16",
            "accumulation": "fp32",
            "custom_hidden": "fp16",
            "custom_final": "fp32",
            "model_path": "fp16 with model-rounding epilogue",
        },
        "kernels": "CUDA-core FP32 FMA after converting FP16 loads. No tensor-core MMA in include/kernels.cuh.",
        "arithmetic_intensity_note": (
            "Weight-dominated GEMV is about 1 FLOP/byte. Published 614 AI TOPS "
            "and RT TFLOPS are not used in this bound."
        ),
        "shapes": [analyze_shape(s) for s in SHAPES],
        "bandwidths": BANDWIDTHS,
    }


def markdown(data: dict) -> str:
    lines = [
        "# Analytical intensity tables",
        "",
        f"Generated by `tools/intensity.py`. Result class: `{data['result_class']}`.",
        "Numbers below are **not** benchmark results.",
        "",
        "| Shape | Hidden | Intermediate | Projection bytes | Custom FFN bytes | FLOP/byte (FFN) | Working set |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in data["shapes"]:
        lines.append(
            f"| {s['name']} | {s['hidden']} | {s['intermediate']} | "
            f"{s['projection']['bytes']} | {s['custom_ffn']['bytes']} | "
            f"{s['custom_ffn']['flops_per_byte']:.3f} | {s['single_layer_working_set_bytes']} |"
        )
    lines += [
        "",
        "100% peak-bandwidth lower bounds (microseconds). Unreachable if efficiency is below 100%, and they ignore launch cost:",
        "",
        "| Shape | Operation | 384 GB/s laptop (NVIDIA published) | 448 GB/s desktop derived |",
        "|---|---|---:|---:|",
    ]
    for s in data["shapes"]:
        b384, b448 = s["bandwidth_lower_bounds_us"]
        lines.append(
            f"| {s['name']} | projection | {b384['projection_us_if_100pct_peak']:.3f} | "
            f"{b448['projection_us_if_100pct_peak']:.3f} |"
        )
        lines.append(
            f"| {s['name']} | custom SwiGLU | {b384['custom_ffn_us_if_100pct_peak']:.3f} | "
            f"{b448['custom_ffn_us_if_100pct_peak']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "docs/rtx-5060-bounds.json")
    p.add_argument("--markdown", type=Path, default=ROOT / "docs/rtx-5060-bounds.md")
    a = p.parse_args()
    data = report()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(data, indent=2) + "\n")
    text = markdown(data)
    if a.markdown:
        a.markdown.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
