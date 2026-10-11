"""Repeatable GPU benchmark sweep. Requires a built decode_bench."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary",type=Path,default=root/"build/decode_bench")
    p.add_argument("--output",type=Path,default=root/"results/sweep")
    p.add_argument("--model",action="store_true")
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    logs=[]
    for seed in (17,101,997):
        for suite in ("correctness","decode"):
            out=a.output/f"{suite}-{seed}.json"
            cmd=[str(a.binary.resolve()),"--suite",suite,"--seed",str(seed),"--output",str(out)]
            run=subprocess.run(cmd,text=True,capture_output=True,timeout=900)
            logs.append({"command":cmd,"exit_code":run.returncode,"stderr":run.stderr})
            (a.output/"commands.json").write_text(json.dumps(logs,indent=2)+"\n")
            if run.returncode:
                raise SystemExit(f"GPU gate failed: {suite}, seed={seed}; see commands.json")
            subprocess.run([sys.executable,str(root/"tools/report.py"),str(out),"--output",str(out.with_suffix(".md"))],check=True)
    if a.model:
        out=a.output/"model.json"
        subprocess.run([sys.executable,str(root/"tools/model_bench.py"),"--output",str(out)],check=True,timeout=900)
        subprocess.run([sys.executable,str(root/"tools/model_report.py"),str(out),"--output",str(out.with_suffix(".md"))],check=True)

if __name__=="__main__":
    main()
