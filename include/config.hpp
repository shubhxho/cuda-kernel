#pragma once
#include <array>
#include <stdexcept>
namespace decode {
struct Config { int id, warps, rows, pack; };
// One list generates both portable metadata and CUDA specializations.
#define DECODE_CONFIGS(X) \
    X(1,1,4,2) X(2,2,4,2) X(4,4,4,2) X(8,8,4,2) \
    X(16,1,4,8) X(17,2,4,8) X(18,4,2,8) X(19,1,8,8) X(20,2,2,8) X(21,4,1,8)
#define DECODE_CONFIG_VALUE(ID,WARPS,ROWS,PACK) {ID,WARPS,ROWS,PACK},
inline constexpr Config configs[] = {DECODE_CONFIGS(DECODE_CONFIG_VALUE)};
#undef DECODE_CONFIG_VALUE
inline bool valid_config(int id) {
    for(auto c:configs) if(c.id==id) return true;
    return false;
}
inline Config config(int id) {
    for(auto c:configs) if(c.id==id) return c;
    throw std::invalid_argument("unknown kernel configuration");
}
}
