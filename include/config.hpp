#pragma once
#include <array>
#include <stdexcept>
namespace decode {
struct Config { int id, warps, rows, pack; };
inline constexpr std::array<Config,4> configs{{{1,1,4,2},{2,2,4,2},{4,4,4,2},{8,8,4,2}}};
inline bool valid_config(int id) {
    for(auto c:configs) if(c.id==id) return true;
    return false;
}
inline Config config(int id) {
    for(auto c:configs) if(c.id==id) return c;
    throw std::invalid_argument("unknown kernel configuration");
}
}
