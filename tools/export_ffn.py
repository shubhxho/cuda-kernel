"""Export one captured SwiGLU layer, without downloading a model or executing pickle."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    import numpy as np
    from safetensors import safe_open

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True, help="local safetensors shard containing this layer")
    p.add_argument("--prefix", required=True, help="e.g. model.layers.0.mlp")
    p.add_argument("--input", type=Path, required=True, help="captured post-norm FFN input, one token .npy")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with safe_open(args.weights, framework="numpy") as f:
        arrays = {}
        for short, key in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
            # BF16 safetensors require a conversion before this numpy exporter.
            arrays[short] = f.get_tensor(f"{args.prefix}.{key}.weight")
            bias_key = f"{args.prefix}.{key}.bias"
            arrays[short + "_bias"] = f.get_tensor(bias_key) if bias_key in f.keys() else np.zeros(arrays[short].shape[0])
    arrays["input"] = np.load(args.input, allow_pickle=False).reshape(-1)
    intermediate, hidden = arrays["gate"].shape
    expected = {"gate": (intermediate, hidden), "up": (intermediate, hidden),
                "down": (hidden, intermediate), "input": (hidden,),
                "gate_bias": (intermediate,), "up_bias": (intermediate,), "down_bias": (hidden,)}
    hashes = {}
    for name, array in arrays.items():
        if array.shape != expected[name] or not np.isfinite(array).all():
            raise ValueError(f"invalid {name} tensor: {array.shape}")
        array = np.asarray(array, dtype="<f2", order="C")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} overflows FP16")
        data = array.tobytes(order="C")
        (args.output / f"{name}.f16").write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    manifest = {"hidden": hidden, "intermediate": intermediate, "weights": str(args.weights),
                "prefix": args.prefix, "input": str(args.input), "sha256": hashes,
                "precision": "fp16-storage-fp32-accumulation"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"--fixture {args.output} --hidden {hidden} --intermediate {intermediate}")


if __name__ == "__main__":
    main()
