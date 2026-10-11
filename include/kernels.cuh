#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace decode {
// Each row gets WPR warps. Half2 loads coalesce; odd strides use scalar loads.
// All lanes participate in reductions, including rows outside the output tail.
template<int WPR, bool GATED, bool RESIDUAL, bool HALF_OUT = false, bool MODEL_ROUND = false>
__global__ void project(const half* __restrict__ w, const half* __restrict__ u,
                        const half* __restrict__ x, const half* __restrict__ bias,
                        const half* __restrict__ ubias, const half* __restrict__ residual,
                        float* __restrict__ y, half* __restrict__ h, int rows, int cols) {
    constexpr int ROWS = 4;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int group = warp / WPR;
    int subwarp = warp % WPR;
    int row = blockIdx.x * ROWS + group;
    int t = subwarp*32 + lane;
    float a = 0, b = 0;
    if (row < rows) {
        size_t offset = size_t(row)*cols;
        if ((cols & 1) == 0) {
            auto ww = reinterpret_cast<const half2*>(w + offset);
            auto xx = reinterpret_cast<const half2*>(x);
            for (int c = t; c < cols/2; c += 32*WPR) {
                float2 wf = __half22float2(ww[c]), xf = __half22float2(xx[c]);
                a = fmaf(wf.x, xf.x, a); a = fmaf(wf.y, xf.y, a);
                if constexpr (GATED) {
                    float2 uf = __half22float2(reinterpret_cast<const half2*>(u + offset)[c]);
                    b = fmaf(uf.x, xf.x, b); b = fmaf(uf.y, xf.y, b);
                }
            }
        } else {
            for (int c = t; c < cols; c += 32*WPR) {
                float xf = __half2float(x[c]);
                a = fmaf(__half2float(w[offset+c]), xf, a);
                if constexpr (GATED) b = fmaf(__half2float(u[offset+c]), xf, b);
            }
        }
    }
    for (int d = 16; d; d >>= 1) {
        a += __shfl_down_sync(0xffffffff, a, d);
        if constexpr (GATED) b += __shfl_down_sync(0xffffffff, b, d);
    }
    __shared__ float partial_a[ROWS*WPR];
    __shared__ float partial_b[ROWS*WPR];
    if constexpr (WPR > 1) {
        if (lane == 0) { partial_a[warp] = a; if constexpr (GATED) partial_b[warp] = b; }
        __syncthreads();
        if (subwarp == 0 && lane == 0) {
            a = 0; b = 0;
            #pragma unroll
            for (int i = 0; i < WPR; ++i) {
                a += partial_a[group*WPR+i];
                if constexpr (GATED) b += partial_b[group*WPR+i];
            }
        }
    }
    if (row < rows && subwarp == 0 && lane == 0) {
        if (bias) a += __half2float(bias[row]);
        if constexpr (GATED) {
            if (ubias) b += __half2float(ubias[row]);
            if constexpr (MODEL_ROUND) {
                // Match eager FP16 Linear -> SiLU -> multiply rounding boundaries.
                a = __half2float(__float2half_rn(a)); b = __half2float(__float2half_rn(b));
                float activated = __half2float(__float2half_rn(a / (1.0f + expf(-a))));
                h[row] = __float2half_rn(activated*b);
            } else h[row] = __float2half_rn((a / (1.0f + expf(-a))) * b);
        } else {
            if constexpr (RESIDUAL) a += __half2float(residual[row]);
            if constexpr (HALF_OUT) h[row] = __float2half_rn(a);
            else y[row] = a;
        }
    }
}
template<bool GATED, bool RESIDUAL, bool HALF_OUT = false, bool MODEL_ROUND = false>
inline void launch(int wpr, const half* w, const half* u, const half* x,
                   const half* bias, const half* ubias, const half* residual,
                   float* y, half* h, int rows, int cols, cudaStream_t stream) {
    #define DISPATCH(N) case N: project<N,GATED,RESIDUAL,HALF_OUT,MODEL_ROUND><<<(rows+3)/4,128*N,0,stream>>>(w,u,x,bias,ubias,residual,y,h,rows,cols); break
    switch (wpr) { DISPATCH(1); DISPATCH(2); DISPATCH(4); DISPATCH(8); default: throw std::invalid_argument("unknown kernel configuration"); }
    #undef DISPATCH
}
__global__ void gate_epilogue(const float* a, const float* b, const half* bias,
                              const half* ubias, half* h, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        float g = a[i]+__half2float(bias[i]), v = b[i]+__half2float(ubias[i]);
        h[i] = __float2half_rn((g/(1.0f+expf(-g)))*v);
    }
}
__global__ void residual_epilogue(float* y, const half* bias, const half* x, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) y[i] += __half2float(bias[i]) + __half2float(x[i]);
}
}
