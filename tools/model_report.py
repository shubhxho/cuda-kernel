"""Render measured full-model results; reject incomplete or incorrect evidence."""
import argparse
import json
import math
from pathlib import Path
import statistics


def render(data):
    if data.get("kind") != "pretrained-model-decode" or not data.get("resolved_revision"):
        raise ValueError("missing pretrained model identity")
    if not data.get("cases") or data.get("patched_layers", 0) <= 0:
        raise ValueError("no model measurements")
    lines = [f"Model: `{data['model']}` at `{data['resolved_revision']}` on {data['gpu']}.",
             f"FP16, batch one, {data['patched_layers']} custom MLP layers; SDPA attention in both paths.",
             "Prefill, model loading, and extension compilation are excluded. Timed decode uses the same growing KV cache.", "",
             "| Prompt | Stock ms/token | Custom ms/token | Wall ratio | Result |",
             "|---|---:|---:|---:|---|"]
    for c in data["cases"]:
        if c.get("exact_greedy_match") is not True or c.get("logits_close") is not True or not math.isfinite(c["max_logit_abs_error"]):
            raise ValueError("model correctness failed")
        if not math.isfinite(c["max_logit_scaled_error"]) or not 0 <= c["max_logit_scaled_error"] <= 1:
            raise ValueError("logit tolerance failed")
        for v in (c["tokens"], data["patched_layers"]):
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError("invalid model workload counts")
        expected_calls = c["tokens"] * data["patched_layers"]
        for side in ("custom", "baseline"):
            samples = c[side]["samples"]
            if len(samples) < 3:
                raise ValueError("insufficient model samples")
            for s in samples:
                for key in ("wall_ms_per_token", "gpu_ms_per_token"):
                    if not math.isfinite(s[key]) or s[key] <= 0:
                        raise ValueError("invalid model timing")
                if s["calls"] != (expected_calls if side == "custom" else 0):
                    raise ValueError("incorrect custom routing")
            for key, sample_key in (("wall_median_ms", "wall_ms_per_token"), ("gpu_median_ms", "gpu_ms_per_token")):
                if not math.isclose(c[side][key], statistics.median(s[sample_key] for s in samples), rel_tol=1e-7):
                    raise ValueError("model medians do not match samples")
        ratio = c["baseline"]["wall_median_ms"] / c["custom"]["wall_median_ms"]
        if not math.isclose(ratio, c["wall_speedup"], rel_tol=1e-7):
            raise ValueError("incorrect model speedup")
        win = ratio >= 1.05 and max(s["wall_ms_per_token"] for s in c["custom"]["samples"]) < min(
            s["wall_ms_per_token"] for s in c["baseline"]["samples"])
        verdict = "separated win" if win else "slower" if ratio < 1 else "inconclusive"
        prompt = c["prompt"].replace("|", "\\|").replace("\n", " ")[:55]
        lines.append(f"| {prompt} | {c['baseline']['wall_median_ms']:.3f} | "
                     f"{c['custom']['wall_median_ms']:.3f} | {ratio:.3f}× | {verdict} |")
    lines += ["", "Every timed decode MLP must use the custom path. Greedy tokens must match exactly; logits must pass the runner's tolerance.",
              "This compares stock Hugging Face/PyTorch eager decoding, not TensorRT-LLM or cuDNN.",
              "Kernel wins can disappear at model level because attention, the KV cache, and Python also consume time."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    text = render(json.loads(a.input.read_text()))
    if a.output:
        a.output.write_text(text)
    else:
        print(text, end="")
