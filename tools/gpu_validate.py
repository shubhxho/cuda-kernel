"""Run on CUDA before accepting any optimization as device-correct."""
from pathlib import Path
import os

def main():
    import torch
    from torch.utils.cpp_extension import load
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required; validation was not executed")
    root=Path(__file__).resolve().parents[1]
    os.environ.setdefault("MAX_JOBS","2")
    ext=load(name="decode_swiglu",sources=[str(root/"src/torch_extension.cu")],
             extra_include_paths=[str(root/"include")],extra_cuda_cflags=["-O3","-lineinfo"])
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.manual_seed(101)
    tested=0
    with torch.inference_mode():
        for hidden,intermediate in [(1,1),(7,9),(8,8),(31,33),(64,70),(256,264),(513,1025)]:
            weights=[torch.randn((intermediate,hidden),device="cuda",dtype=torch.float16)/(hidden**.5),
                     torch.randn((intermediate,hidden),device="cuda",dtype=torch.float16)/(hidden**.5),
                     torch.randn((hidden,intermediate),device="cuda",dtype=torch.float16)/(intermediate**.5)]
            for offset in (0,1):
                backing=torch.randn((hidden+offset,),device="cuda",dtype=torch.float16)
                x=backing[offset:].view(1,1,hidden)
                expected=(torch.nn.functional.silu(x@weights[0].T)*(x@weights[1].T))@weights[2].T
                h=torch.empty((intermediate,),device="cuda",dtype=torch.float16)
                y=torch.empty_like(x)
                for gate in ext.config_ids():
                    for down in ext.config_ids():
                        ext.swiglu_out(x,*weights,h,y,gate,down)
                        torch.testing.assert_close(y,expected,atol=.01,rtol=.02)
                        tested+=1
                # Output is allowed to be reused on one stream, but cannot overwrite input.
                try:
                    ext.swiglu_out(x,*weights,h,x,1,1)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("aliased output was accepted")
                for invalid in (0,999,2**32+1):
                    try:
                        ext.swiglu_out(x,*weights,h,y,invalid,1)
                    except RuntimeError:
                        pass
                    else:
                        raise AssertionError("invalid configuration was accepted")
        torch.cuda.synchronize()
    print(f"{tested} GPU configuration/alignment cases passed")

if __name__=="__main__":
    main()
