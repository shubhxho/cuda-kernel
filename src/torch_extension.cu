#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "kernels.cuh"

torch::Tensor swiglu(torch::Tensor x,torch::Tensor gate,torch::Tensor up,torch::Tensor down,int64_t wpr) {
    for(const auto& t:{x,gate,up,down}) {
        TORCH_CHECK(t.is_cuda() && t.scalar_type()==torch::kFloat16 && t.is_contiguous(),
                    "all inputs must be contiguous CUDA FP16 tensors");
        TORCH_CHECK(t.device()==x.device(),"all inputs must be on the same device");
        TORCH_CHECK(!t.requires_grad(),"inference only: disable gradients");
    }
    TORCH_CHECK(gate.dim()==2 && up.dim()==2 && down.dim()==2 && x.dim()>=1,"invalid dimensions");
    int64_t hidden=gate.size(1),intermediate=gate.size(0);
    TORCH_CHECK(hidden>0 && intermediate>0 && hidden<=100000 && intermediate<=100000,"invalid shape");
    TORCH_CHECK(x.numel()==hidden && x.size(-1)==hidden,"only batch-one single-token decode is supported");
    TORCH_CHECK(up.sizes()==gate.sizes() && down.size(0)==hidden && down.size(1)==intermediate,"incompatible weights");
    TORCH_CHECK(wpr==1 || wpr==2 || wpr==4 || wpr==8,"warps-per-row must be 1, 2, 4, or 8");
    c10::cuda::CUDAGuard guard(x.device());
    auto h=torch::empty({intermediate},x.options());
    auto y=torch::empty_like(x);
    auto stream=at::cuda::getCurrentCUDAStream(x.get_device()).stream();
    auto ptr=[](const torch::Tensor& t){return reinterpret_cast<const half*>(t.data_ptr<at::Half>());};
    decode::launch<true,false,false,true>(int(wpr),ptr(gate),ptr(up),ptr(x),nullptr,nullptr,nullptr,
        nullptr,reinterpret_cast<half*>(h.data_ptr<at::Half>()),int(intermediate),int(hidden),stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    decode::launch<false,false,true>(int(wpr),ptr(down),nullptr,ptr(h),nullptr,nullptr,nullptr,
        nullptr,reinterpret_cast<half*>(y.data_ptr<at::Half>()),int(hidden),int(intermediate),stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("swiglu",&swiglu,"Fused single-token FP16 SwiGLU"); }
