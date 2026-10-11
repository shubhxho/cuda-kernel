#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace decode {
struct Shape { std::string name; int hidden; int intermediate; };
inline std::vector<Shape> shapes(const std::string& suite) {
    if (suite == "correctness") return {{"tiny", 1, 1}, {"odd", 37, 65},
        {"tail", 129, 257}, {"aligned", 512, 1024}, {"one-to-tail", 1, 9}, {"half2-min", 2, 3}, {"vector8-min", 8, 8}, {"below-vector8", 7, 9}, {"above-vector8", 9, 7}, {"below-warp", 31, 33}, {"exact-warp", 32, 32}, {"above-warp", 33, 31}, {"half2-warp", 64, 64}, {"half2-tail", 66, 70}, {"vector8-warp", 256, 256}, {"vector8-loop", 264, 272}};
    if (suite == "decode") return {{"decode-1b-shape", 2048, 8192},
        {"decode-7b-shape", 4096, 11008}, {"decode-8b-shape", 4096, 14336}};
    if (suite == "smoke") return {{"smoke", 512, 1024}};
    throw std::invalid_argument("suite must be correctness, smoke, or decode");
}
inline double silu(double x) { if(x>=0) return x/(1.0+std::exp(-x)); double e=std::exp(x); return x*e/(1.0+e); }
// Double reference, independent of CUDA reduction order and cuBLAS.
inline std::vector<float> matvec(const std::vector<float>& w,
                                const std::vector<float>& x, int rows) {
    if (rows <= 0 || x.empty() || w.size() != size_t(rows) * x.size())
        throw std::invalid_argument("invalid row-major matvec dimensions");
    std::vector<float> y(rows);
    for (int r = 0; r < rows; ++r) {
        double sum = 0;
        for (size_t c = 0; c < x.size(); ++c) sum += double(w[size_t(r)*x.size()+c])*x[c];
        y[r] = float(sum);
    }
    return y;
}
struct Error { double max_abs = 0; double max_scaled = 0; bool passed = true; };
inline Error compare(const std::vector<float>& actual, const std::vector<float>& expected,
                     double atol = 0.003, double rtol = 0.01) {
    if (actual.size() != expected.size() || actual.empty())
        throw std::invalid_argument("comparison sizes differ or are empty");
    if(!std::isfinite(atol) || !std::isfinite(rtol) || atol<=0 || rtol<0) throw std::invalid_argument("invalid tolerance");
    Error e;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            e.passed = false; e.max_abs = e.max_scaled = INFINITY; continue;
        }
        const double delta = std::abs(double(actual[i])-expected[i]);
        const double scaled = delta / (atol + rtol*std::abs(expected[i]));
        e.max_abs = std::max(e.max_abs, delta);
        e.max_scaled = std::max(e.max_scaled, scaled);
        e.passed &= scaled <= 1.0;
    }
    return e;
}
inline double median(std::vector<double> xs) {
    if (xs.empty()) throw std::invalid_argument("no timing samples");
    for (double x : xs) if (!std::isfinite(x) || x <= 0) throw std::invalid_argument("invalid timing");
    std::sort(xs.begin(), xs.end());
    size_t n = xs.size();
    return n%2 ? xs[n/2] : (xs[n/2-1]+xs[n/2])/2;
}
inline std::string json_escape(const std::string& s) {
    std::string out;
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += char(c); }
        else if (c < 32) {
            const char* hex = "0123456789abcdef";
            out += "\\u00"; out += hex[c>>4]; out += hex[c&15];
        } else out += char(c);
    }
    return out;
}
}
