"""Benchmark actual pretrained decoding, with correctness gates before timing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ["Explain why the sky looks blue in three sentences.",
           "Write a short Python function that computes the greatest common divisor.",
           "What happens when water freezes? Explain it simply."]


def main():
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.utils.cpp_extension import load

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--revision", default="main")
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--samples", type=int, default=7)
    p.add_argument("--output", type=Path, default=ROOT / "results/model.json")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("NVIDIA CUDA GPU required; CPU timing cannot establish a CUDA win")
    if not 1 <= args.tokens <= 256 or not 3 <= args.samples <= 31:
        raise SystemExit("tokens must be 1..256 and samples 3..31")
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    os.environ.setdefault("MAX_JOBS", "2")
    ext = load(name="decode_swiglu", sources=[str(ROOT / "src/torch_extension.cu")],
               extra_include_paths=[str(ROOT / "include")], extra_cuda_cflags=["-O3", "-lineinfo"], verbose=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.float16,
        attn_implementation="sdpa", trust_remote_code=False, use_safetensors=True).cuda().eval()
    if model.config.model_type != "llama" or model.config.hidden_act != "silu":
        raise SystemExit("only Llama-style SiLU MLP models are supported")
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if not resolved_revision:
        raise RuntimeError("model revision was not resolved; cannot produce reproducible evidence")
    originals, layer_modules = [], []
    for layer in model.model.layers:
        mlp = layer.mlp
        if any(proj.bias is not None for proj in (mlp.gate_proj, mlp.up_proj, mlp.down_proj)):
            raise SystemExit("model benchmark currently supports bias-free MLPs")
        originals.append(mlp.forward)
        layer_modules.append(mlp)
    enabled = False
    selected_wpr = 1
    selected_down = 1
    calls = 0

    for mlp, original in zip(layer_modules, originals):
        weights = (mlp.gate_proj.weight.detach(), mlp.up_proj.weight.detach(), mlp.down_proj.weight.detach())
        scratch = torch.empty((model.config.intermediate_size,), device="cuda", dtype=torch.float16)
        output = torch.empty((1, 1, model.config.hidden_size), device="cuda", dtype=torch.float16)
        def forward(x, weights=weights, original=original, scratch=scratch, output=output):
            nonlocal calls
            if enabled and x.numel() == x.shape[-1]:
                calls += 1
                ext.swiglu_out(x.contiguous(), *weights, scratch, output, selected_wpr, selected_wpr)
                return output
            return original(x)
        mlp.forward = forward

    @torch.inference_mode()
    def decode(ids, collect=False):
        nonlocal calls
        # Prefill is excluded from decode latency, and follows the stock path in both runs.
        initial = model(input_ids=ids, use_cache=True)
        past = initial.past_key_values
        next_token = initial.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        calls_before = calls
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        logits, tokens = [], []
        started = time.perf_counter()
        start.record()
        for _ in range(args.tokens):
            out = model(input_ids=next_token, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_token = out.logits[:, -1:].argmax(-1)
            if collect:
                logits.append(out.logits[:, -1].detach())
                tokens.append(next_token)
        end.record()
        end.synchronize()
        wall_ms = (time.perf_counter()-started)*1000
        return {"gpu_ms_per_token": start.elapsed_time(end)/args.tokens,
                "wall_ms_per_token": wall_ms/args.tokens,
                "calls": calls-calls_before,
                "logits": torch.stack(logits) if collect else None,
                "tokens": torch.cat(tokens, dim=1) if collect else None}

    # Tune only on the first layer and an independent fixed random input.
    tuning = []
    with torch.inference_mode():
        mlp = layer_modules[0]
        tx = torch.randn((1, 1, model.config.hidden_size), device="cuda", dtype=torch.float16)
        expected = originals[0](tx)
        for wpr in ext.config_ids():
            y = ext.swiglu(tx, mlp.gate_proj.weight.detach(), mlp.up_proj.weight.detach(), mlp.down_proj.weight.detach(), wpr)
            torch.testing.assert_close(y, expected, atol=0.01, rtol=0.02)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                ext.swiglu(tx, mlp.gate_proj.weight.detach(), mlp.up_proj.weight.detach(), mlp.down_proj.weight.detach(), wpr)
            for _ in range(20):
                graph.replay()
            samples = []
            for _ in range(5):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(100):
                    graph.replay()
                end.record(); end.synchronize()
                samples.append(start.elapsed_time(end)*10)
            tuning.append({"wpr": wpr, "samples_us": samples})
    selected_wpr = min(tuning, key=lambda c: statistics.median(c["samples_us"]))["wpr"]
    records = []
    for prompt in PROMPTS:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        enabled = False
        baseline = decode(ids, collect=True)
        enabled = True
        custom = decode(ids, collect=True)
        if custom["calls"] != args.tokens * len(layer_modules):
            raise RuntimeError("not every decode MLP used the custom CUDA path")
        torch.testing.assert_close(custom["logits"], baseline["logits"], atol=0.1, rtol=0.02)
        if not torch.equal(custom["tokens"], baseline["tokens"]):
            raise RuntimeError("greedy generation differs; no performance claim allowed")
        errors = (custom["logits"].float()-baseline["logits"].float()).abs()
        scaled_error = errors / (0.1 + 0.02 * baseline["logits"].float().abs())
        collected = {"baseline": [], "custom": []}
        for sample in range(args.samples):
            for side in (("baseline", "custom") if sample % 2 == 0 else ("custom", "baseline")):
                enabled = side == "custom"
                run = decode(ids)
                collected[side].append({k: run[k] for k in ("gpu_ms_per_token", "wall_ms_per_token", "calls")})
        stats = {side: {"samples": values,
                       "gpu_median_ms": statistics.median(v["gpu_ms_per_token"] for v in values),
                       "wall_median_ms": statistics.median(v["wall_ms_per_token"] for v in values)}
                 for side, values in collected.items()}
        speedup = stats["baseline"]["wall_median_ms"] / stats["custom"]["wall_median_ms"]
        records.append({"prompt": prompt, "prompt_tokens": ids.numel(), "tokens": args.tokens,
                        "exact_greedy_match": True, "max_logit_abs_error": errors.max().item(),
                        "logits_close": True, "max_logit_scaled_error": scaled_error.max().item(),
                        "generated_text": tokenizer.decode(custom["tokens"][0]), **stats,
                        "wall_speedup": speedup})
        print(f"{prompt[:35]}: stock {stats['baseline']['wall_median_ms']:.3f} ms/token, "
              f"custom {stats['custom']['wall_median_ms']:.3f} ms/token, {speedup:.3f}x")
    sources = [ROOT / "include/kernels.cuh", ROOT / "src/torch_extension.cu", Path(__file__)]
    result = {"schema_version": 1, "kind": "pretrained-model-decode", "model": args.model,
              "requested_revision": args.revision, "resolved_revision": resolved_revision,
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "cuda": torch.version.cuda, "transformers": transformers.__version__,
              "precision": "fp16", "attention": "sdpa", "batch": 1,
              "timing": "eager-decode-gpu-events-and-wall", "prefill_included": False,
              "patched_layers": len(layer_modules), "wpr": selected_wpr, "tuning": tuning,
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "cases": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
