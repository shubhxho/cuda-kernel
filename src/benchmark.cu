#include "contract.hpp"
#include "kernels.cuh"
#include <cublas_v2.h>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>

using namespace decode;
void ck(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void bk(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS status " + std::to_string(int(status)));
}
template<class T> struct Buffer {
    T* p = nullptr; size_t n;
    explicit Buffer(size_t count) : n(count) { ck(cudaMalloc(reinterpret_cast<void**>(&p), n*sizeof(T))); }
    ~Buffer() { if(p) cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void upload(const std::vector<T>& v) {
        if(v.size()!=n) throw std::runtime_error("upload size mismatch");
        ck(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> v(n); ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost)); return v;
    }
};
struct Context {
    cudaStream_t stream = nullptr; cublasHandle_t blas = nullptr;
    Context() {
        ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        bk(cublasCreate(&blas)); bk(cublasSetStream(blas,stream));
        bk(cublasSetMathMode(blas,CUBLAS_DEFAULT_MATH));
    }
    ~Context() { if(blas) cublasDestroy(blas); if(stream) cudaStreamDestroy(stream); }
};
struct Graph {
    cudaGraph_t graph = nullptr; cudaGraphExec_t exec = nullptr;
    Graph(Context& c, const std::function<void()>& fn) {
        // Warm-up initializes cuBLAS workspaces before graph capture.
        fn(); ck(cudaStreamSynchronize(c.stream));
        ck(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeThreadLocal));
        try { fn(); } catch(...) {
            cudaGraph_t abandoned=nullptr;cudaStreamEndCapture(c.stream,&abandoned);
            if(abandoned) cudaGraphDestroy(abandoned);throw;
        }
        ck(cudaStreamEndCapture(c.stream,&graph));
        auto status=cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0);
        if(status!=cudaSuccess) { cudaGraphDestroy(graph);graph=nullptr;ck(status); }
        ck(cudaGraphUpload(exec,c.stream));ck(cudaStreamSynchronize(c.stream));
    }
    ~Graph() { if(exec) cudaGraphExecDestroy(exec); if(graph) cudaGraphDestroy(graph); }
    Graph(const Graph&) = delete;
    void run(Context& c) { ck(cudaGraphLaunch(exec,c.stream)); }
};
struct Event {
    cudaEvent_t event = nullptr;
    Event() { ck(cudaEventCreate(&event)); }
    ~Event() { if(event) cudaEventDestroy(event); }
};
std::vector<double> timings(Context& c, Graph& g, int repeats, int samples) {
    for(int i=0;i<20;++i) g.run(c);
    ck(cudaStreamSynchronize(c.stream));
    Event a,b; std::vector<double> out;
    for(int s=0;s<samples;++s) {
        ck(cudaEventRecord(a.event,c.stream));
        for(int i=0;i<repeats;++i) g.run(c);
        ck(cudaEventRecord(b.event,c.stream)); ck(cudaEventSynchronize(b.event));
        float ms=0; ck(cudaEventElapsedTime(&ms,a.event,b.event));
        out.push_back(double(ms)*1000/repeats);
    }
    return out;
}
std::vector<float> to_float(const std::vector<half>& v) {
    std::vector<float> f(v.size());
    for(size_t i=0;i<v.size();++i) f[i]=__half2float(v[i]);
    return f;
}
std::vector<half> random_half(size_t n, float scale, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-scale,scale);
    std::vector<half> v(n);
    for(auto& x:v) x=__float2half_rn(dist(rng));
    return v;
}
std::vector<half> load_half(const std::string& directory,const std::string& name,size_t n) {
    std::ifstream file(directory+"/"+name,std::ios::binary|std::ios::ate);
    if(!file || file.tellg()!=std::streamoff(n*sizeof(half)))
        throw std::runtime_error("missing or incorrectly sized fixture: "+name);
    file.seekg(0);std::vector<half> v(n);
    file.read(reinterpret_cast<char*>(v.data()),std::streamsize(n*sizeof(half)));
    if(!file) throw std::runtime_error("fixture read failed: "+name);
    for(auto h:v) if(!std::isfinite(__half2float(h)))
        throw std::runtime_error("non-finite fixture: "+name);
    return v;
}
struct Workload {
    int hidden, intermediate;
    Buffer<half> wg,wu,wd,x,bg,bu,bd,h;
    Buffer<float> g,u,y;
    std::vector<float> projection_ref,ffn_ref;
    Workload(const Shape& s, unsigned seed, const std::string& fixture) : hidden(s.hidden),intermediate(s.intermediate),
        wg(size_t(hidden)*intermediate),wu(size_t(hidden)*intermediate),
        wd(size_t(hidden)*intermediate),x(hidden),bg(intermediate),bu(intermediate),
        bd(hidden),h(intermediate),g(intermediate),u(intermediate),y(hidden) {
        std::mt19937 rng(seed);
        auto hwg=random_half(wg.n,1/std::sqrt(float(hidden)),rng);
        auto hwu=random_half(wu.n,1/std::sqrt(float(hidden)),rng);
        auto hwd=random_half(wd.n,1/std::sqrt(float(intermediate)),rng);
        auto hx=random_half(x.n,1,rng), hbg=random_half(bg.n,.1f,rng);
        auto hbu=random_half(bu.n,.1f,rng), hbd=random_half(bd.n,.1f,rng);
        if(!fixture.empty()) {
            hwg=load_half(fixture,"gate.f16",wg.n);hwu=load_half(fixture,"up.f16",wu.n);
            hwd=load_half(fixture,"down.f16",wd.n);hx=load_half(fixture,"input.f16",x.n);
            hbg=load_half(fixture,"gate_bias.f16",bg.n);hbu=load_half(fixture,"up_bias.f16",bu.n);
            hbd=load_half(fixture,"down_bias.f16",bd.n);
        }
        wg.upload(hwg);wu.upload(hwu);wd.upload(hwd);x.upload(hx);
        bg.upload(hbg);bu.upload(hbu);bd.upload(hbd);
        auto fx=to_float(hx);
        projection_ref=matvec(to_float(hwg),fx,intermediate);
        auto up=matvec(to_float(hwu),fx,intermediate);
        std::vector<float> activation(intermediate);
        for(int i=0;i<intermediate;++i) {
            double v=silu(double(projection_ref[i])+__half2float(hbg[i])) *
                (double(up[i])+__half2float(hbu[i]));
            activation[i]=__half2float(__float2half_rn(float(v)));
        }
        ffn_ref=matvec(to_float(hwd),activation,hidden);
        for(int i=0;i<hidden;++i) ffn_ref[i]+=fx[i]+__half2float(hbd[i]);
    }
    void gemv(Context& c,const half* w,const half* input,float* out,int rows,int cols,int algo) {
        float alpha=1,beta=0;
        // Row-major W[rows,cols] is column-major [cols,rows], transposed by cuBLAS.
        bk(cublasGemmEx(c.blas,CUBLAS_OP_T,CUBLAS_OP_N,rows,1,cols,
            &alpha,w,CUDA_R_16F,cols,input,CUDA_R_16F,cols,
            &beta,out,CUDA_R_32F,rows,CUBLAS_COMPUTE_32F,
            algo==0 ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    void baseline(Context& c,bool ffn,int algo) {
        gemv(c,wg.p,x.p,g.p,intermediate,hidden,algo);
        if(ffn) {
            gemv(c,wu.p,x.p,u.p,intermediate,hidden,algo);
            gate_epilogue<<<(intermediate+255)/256,256,0,c.stream>>>(g.p,u.p,bg.p,bu.p,h.p,intermediate);
            gemv(c,wd.p,h.p,y.p,hidden,intermediate,algo);
            residual_epilogue<<<(hidden+255)/256,256,0,c.stream>>>(y.p,bd.p,x.p,hidden);
        }
        ck(cudaPeekAtLastError());
    }
    void custom(Context& c,bool ffn,int wpr,int down_config=-1) {
        if(ffn) {
            launch<true,false>(wpr,wg.p,wu.p,x.p,bg.p,bu.p,nullptr,nullptr,h.p,intermediate,hidden,c.stream);
            launch<false,true>(down_config<0?wpr:down_config,wd.p,nullptr,h.p,bd.p,nullptr,x.p,y.p,nullptr,hidden,intermediate,c.stream);
        } else launch<false,false>(wpr,wg.p,nullptr,x.p,nullptr,nullptr,nullptr,g.p,nullptr,intermediate,hidden,c.stream);
        ck(cudaPeekAtLastError());
    }
    Error validate(Context& c,bool ffn) {
        ck(cudaStreamSynchronize(c.stream));
        return compare(ffn ? y.download() : g.download(),ffn ? ffn_ref : projection_ref);
    }
};
struct Candidate {
    std::string name; Error error; std::vector<double> tuning,measured;
    std::unique_ptr<Graph> graph;
};
std::string array_json(const std::vector<double>& xs) {
    std::ostringstream o; o<<'[';
    for(size_t i=0;i<xs.size();++i) { if(i) o<<','; o<<std::setprecision(10)<<xs[i]; }
    o<<']'; return o.str();
}
std::string candidate_json(const Candidate& c) {
    std::ostringstream o; o<<std::setprecision(10);
    o<<"{\"name\":\""<<c.name<<"\",\"correct\":true,\"max_abs_error\":"<<c.error.max_abs
     <<",\"max_scaled_error\":"<<c.error.max_scaled<<",\"tuning_us\":"<<array_json(c.tuning)
     <<",\"samples_us\":"<<array_json(c.measured)<<",\"median_us\":"<<median(c.measured)<<'}';
    return o.str();
}
std::string run_case(Context& ctx,Workload& w,const Shape& shape,bool ffn,int repeats,int samples) {
    std::vector<Candidate> custom,baseline;
    for(int algo:{0,1}) {
        Candidate c; c.name=algo==0?"cublas-default":"cublas-tensor-op";
        auto fn=[&ctx,&w,ffn,algo]{w.baseline(ctx,ffn,algo);};
        fn();c.error=w.validate(ctx,ffn);
        if(!c.error.passed) throw std::runtime_error(c.name+" correctness failed for "+shape.name);
        c.graph=std::make_unique<Graph>(ctx,fn);
        c.graph->run(ctx);c.error=w.validate(ctx,ffn);
        if(!c.error.passed) throw std::runtime_error("captured candidate correctness failed: "+c.name);
        c.tuning=timings(ctx,*c.graph,repeats,3);baseline.push_back(std::move(c));
    }
    for(auto gate_config:configs) for(auto down:configs) {
        if(!ffn && down.id!=1) continue;
        int wpr=gate_config.id,down_id=down.id;
        Candidate c; c.name="custom-g"+std::to_string(wpr)+"-d"+std::to_string(down_id);
        auto fn=[&ctx,&w,ffn,wpr,down_id]{w.custom(ctx,ffn,wpr,down_id);};
        fn();c.error=w.validate(ctx,ffn);
        if(!c.error.passed) throw std::runtime_error(c.name+" correctness failed for "+shape.name);
        c.graph=std::make_unique<Graph>(ctx,fn);
        c.graph->run(ctx);c.error=w.validate(ctx,ffn);
        if(!c.error.passed) throw std::runtime_error("captured candidate correctness failed: "+c.name);
        c.tuning=timings(ctx,*c.graph,repeats,3);custom.push_back(std::move(c));
    }
    auto best=[](std::vector<Candidate>& cs)->Candidate& {
        return *std::min_element(cs.begin(),cs.end(),[](const auto& a,const auto& b){
            return median(a.tuning)<median(b.tuning);
        });
    };
    auto& a=best(custom);auto& b=best(baseline);
    // Fresh, interleaved holdout measurements; tuning samples never establish wins.
    for(int s=0;s<samples;++s) {
        auto measure=[&](Candidate& c){ c.measured.push_back(timings(ctx,*c.graph,repeats,1)[0]); };
        if(s%2) {measure(a);measure(b);} else {measure(b);measure(a);}
    }
    // Verify the captured paths too, including any library capture-specific behavior.
    a.graph->run(ctx); a.error=w.validate(ctx,ffn);
    b.graph->run(ctx); b.error=w.validate(ctx,ffn);
    if(!a.error.passed || !b.error.passed) throw std::runtime_error("captured graph correctness failed");
    double ratio=median(b.measured)/median(a.measured);
    std::cerr<<shape.name<<' '<<(ffn?"swiglu-ffn":"projection")<<": custom "<<median(a.measured)
             <<" us, cuBLAS "<<median(b.measured)<<" us, ratio "<<ratio<<"x\n";
    std::ostringstream out;out<<std::setprecision(10);
    out<<"{\"shape\":\""<<shape.name<<"\",\"operation\":\""<<(ffn?"swiglu-ffn":"projection")
       <<"\",\"hidden\":"<<w.hidden<<",\"intermediate\":"<<w.intermediate
       <<",\"custom\":"<<candidate_json(a)<<",\"baseline\":"<<candidate_json(b)
       <<",\"speedup\":"<<ratio<<",\"candidates\":[";
    bool first=true;
    for(auto* cs:{&baseline,&custom}) for(auto& c:*cs) {
        if(!first) out<<',';first=false;
        out<<"{\"name\":\""<<c.name<<"\",\"correct\":true,\"tuning_us\":"<<array_json(c.tuning)<<'}';
    }
    out<<"]}";return out.str();
}
int integer(const std::string& v) {
    size_t n=0;int x=std::stoi(v,&n);
    if(n!=v.size() || x<=0 || x>100000) throw std::invalid_argument("invalid positive bounded integer");
    return x;
}
int main(int argc,char** argv) {
    try {
        std::string suite="decode",output,fixture;int repeats=100,samples=9,seed=17,hidden=0,intermediate=0;
        for(int i=1;i<argc;++i) {
            std::string arg=argv[i];
            if(arg=="--help") {
                std::cout<<"decode_bench [--suite decode|smoke|correctness] [--repeats 100] [--samples 9] [--seed 17] [--output result.json]\n"
                         <<"  --fixture DIR --hidden H --intermediate I : captured model FFN and input\n";
                return 0;
            }
            if(i+1>=argc) throw std::invalid_argument("missing argument value");
            std::string value=argv[++i];
            if(arg=="--suite") suite=value;
            else if(arg=="--output") output=value;
            else if(arg=="--repeats") repeats=integer(value);
            else if(arg=="--samples") samples=integer(value);
            else if(arg=="--seed") seed=integer(value);
            else if(arg=="--fixture") fixture=value;
            else if(arg=="--hidden") hidden=integer(value);
            else if(arg=="--intermediate") intermediate=integer(value);
            else throw std::invalid_argument("unknown option: "+arg);
        }
        if(samples<3) throw std::invalid_argument("at least 3 holdout samples required");
        auto cases=shapes(suite);
        if(!fixture.empty()) {
            if(!hidden || !intermediate) throw std::invalid_argument("fixture needs --hidden and --intermediate");
            cases={{"captured-model-ffn",hidden,intermediate}};
        } else if(hidden || intermediate) throw std::invalid_argument("custom dimensions require --fixture");
        int count=0;ck(cudaGetDeviceCount(&count));
        if(!count) throw std::runtime_error("no NVIDIA CUDA device available");
        ck(cudaSetDevice(0));cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,0));
        Context ctx;int blas_version=0,driver=0,runtime=0;
        bk(cublasGetVersion(ctx.blas,&blas_version));ck(cudaDriverGetVersion(&driver));ck(cudaRuntimeGetVersion(&runtime));
        std::ostringstream result;
        result<<"{\"schema_version\":1,\"gpu\":\""<<json_escape(prop.name)
              <<"\",\"compute_capability\":\""<<prop.major<<'.'<<prop.minor<<"\",\"l2_bytes\":"<<prop.l2CacheSize
              <<",\"driver_version\":"<<driver<<",\"runtime_version\":"<<runtime
              <<",\"cublas_version\":"<<blas_version<<",\"suite\":\""<<suite
              <<"\",\"seed\":"<<seed<<",\"repeats\":"<<repeats<<",\"samples\":"<<samples
              <<",\"timing\":\"cuda-graph-events\",\"cache_policy\":\"resident-weights-warm\""
              <<",\"data\":\""<<(fixture.empty()?"synthetic-fp16-model-shapes":"captured-model-fixture")
              <<"\",\"atol\":0.003,\"rtol\":0.01,\"cases\":[";
        bool first=true;
        for(const auto& shape:cases) {
            Workload w(shape,unsigned(seed),fixture);
            for(bool ffn:{false,true}) {
                if(!first) result<<',';first=false;
                result<<run_case(ctx,w,shape,ffn,repeats,samples);
            }
        }
        result<<"]}\n";
        if(output.empty()) std::cout<<result.str();
        else {
            std::ofstream file(output);
            if(!file) throw std::runtime_error("cannot open output file: "+output);
            file<<result.str();file.close();
            if(!file) throw std::runtime_error("result write failed");
        }
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;
    }
}
