#include "contract.hpp"
#include <iostream>
#include <limits>

void require(bool value) { if (!value) throw std::runtime_error("test failed"); }
int main() {
    using namespace decode;
    require(matvec({1,2,3,4,5,6}, {2,-1,3}, 2) == std::vector<float>({9,21}));
    require(silu(0) == 0);
    require(std::abs(silu(2)-1.7615941559557646) < 1e-12);
    require(compare({1.001f,0}, {1,0}).passed);
    require(!compare({1.2f}, {1}).passed);
    require(!compare({std::numeric_limits<float>::quiet_NaN()}, {0}).passed);
    require(median({4,1,2,3}) == 2.5);
    require(median({4,1,2}) == 2);
    require(shapes("correctness")[1].hidden == 37);
    require(json_escape("a\"\n\\") == "a\\\"\\u000a\\\\");
    require(std::isfinite(silu(-1000)) && silu(-1000)==0);
    require(silu(1000)==1000);
    require(!compare({INFINITY}, {INFINITY}).passed);
    { bool failed=false;try { compare({1},{1},.003,-.01); } catch(const std::invalid_argument&) { failed=true; } require(failed); }
    int rejected = 0;
    try { matvec({1}, {1,2}, 1); } catch (const std::invalid_argument&) { ++rejected; }
    try { shapes("typo"); } catch (const std::invalid_argument&) { ++rejected; }
    try { median({0}); } catch (const std::invalid_argument&) { ++rejected; }
    try { compare({}, {}); } catch (const std::invalid_argument&) { ++rejected; }
    require(rejected == 4);
    std::cout << "host reference, numeric gates, shapes, and JSON contracts passed\n";
}
