#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "kernels.cuh"

void swiglu_out(torch::Tensor x,torch::Tensor gate,torch::Tensor up,torch::Tensor down,
                torch::Tensor h,torch::Tensor y,int64_t gate_config,int64_t down_config) {
    for(const auto& t:{x,gate,up,down,h,y}) {
        TORCH_CHECK(t.is_cuda() && t.scalar_type()==torch::kFloat16 && t.is_contiguous(),
                    "all buffers must be contiguous CUDA FP16 tensors");
        TORCH_CHECK(t.device()==x.device(),"buffers must share a device");
        TORCH_CHECK(!t.requires_grad(),"inference only: disable gradients");
    }
    TORCH_CHECK(gate.dim()==2 && up.dim()==2 && down.dim()==2 && x.dim()>=1,"invalid dimensions");
    int64_t hidden=gate.size(1),intermediate=gate.size(0);
    TORCH_CHECK(hidden>0 && intermediate>0 && hidden<=100000 && intermediate<=100000,"invalid shape");
    TORCH_CHECK(x.numel()==hidden && x.size(-1)==hidden,"single-token batch-one decode required");
    TORCH_CHECK(up.sizes()==gate.sizes() && down.size(0)==hidden && down.size(1)==intermediate,"incompatible weights");
    TORCH_CHECK(h.numel()==intermediate && y.sizes()==x.sizes(),"invalid workspace shape");
    TORCH_CHECK(gate_config>=0 && gate_config<=1000 && down_config>=0 && down_config<=1000,"configuration out of range");
    TORCH_CHECK(decode::valid_config(int(gate_config)) && decode::valid_config(int(down_config)),"invalid configuration");
    auto overlaps=[](const torch::Tensor& a,const torch::Tensor& b) {
        auto ap=reinterpret_cast<uintptr_t>(a.data_ptr()),bp=reinterpret_cast<uintptr_t>(b.data_ptr());
        return ap<bp+b.numel()*b.element_size() && bp<ap+a.numel()*a.element_size();
    };
    for(const auto& t:{x,gate,up,down}) {
        TORCH_CHECK(!overlaps(h,t) && !overlaps(y,t),"workspace overlaps input storage");
    }
    TORCH_CHECK(!overlaps(h,y),"scratch and output overlap");
    c10::cuda::CUDAGuard guard(x.device());
    auto stream=at::cuda::getCurrentCUDAStream(x.get_device()).stream();
    auto ptr=[](const torch::Tensor& t){return reinterpret_cast<const half*>(t.data_ptr<at::Half>());};
    decode::launch<true,false,false,true>(int(gate_config),ptr(gate),ptr(up),ptr(x),nullptr,nullptr,nullptr,
        nullptr,reinterpret_cast<half*>(h.data_ptr<at::Half>()),int(intermediate),int(hidden),stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    decode::launch<false,false,true>(int(down_config),ptr(down),nullptr,ptr(h),nullptr,nullptr,nullptr,
        nullptr,reinterpret_cast<half*>(y.data_ptr<at::Half>()),int(hidden),int(intermediate),stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
torch::Tensor swiglu(torch::Tensor x,torch::Tensor gate,torch::Tensor up,torch::Tensor down,
                     int64_t gate_config,int64_t down_config=-1) {
    TORCH_CHECK(gate.dim()==2,"gate must be a matrix");
    auto h=torch::empty({gate.size(0)},x.options()),y=torch::empty_like(x);
    swiglu_out(x,gate,up,down,h,y,gate_config,down_config<0?gate_config:down_config);
    return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("swiglu",&swiglu,pybind11::arg("x"),pybind11::arg("gate"),pybind11::arg("up"),pybind11::arg("down"),
          pybind11::arg("gate_config"),pybind11::arg("down_config")=-1);
    m.def("swiglu_out",&swiglu_out);
    m.def("config_ids",[]{std::vector<int> ids;for(auto c:decode::configs) ids.push_back(c.id);return ids;});
}
