#include "config.hpp"
#include "contract.hpp"
#include <iostream>

// Portable address-coverage audit. This does not execute CUDA instructions.
void audit(decode::Config c, int rows, int cols, int input_offset_bytes) {
    const int pack = c.pack==8 && cols%8==0 && input_offset_bytes%16==0 ? 8 :
                     cols%2==0 && input_offset_bytes%4==0 ? 2 : 1;
    std::vector<int> columns(cols,0),outputs(rows,0);
    for(int thread=0;thread<32*c.warps;++thread)
        for(int unit=thread;unit<cols/pack;unit+=32*c.warps)
            for(int element=0;element<pack;++element) ++columns[unit*pack+element];
    for(int block=0;block<(rows+c.rows-1)/c.rows;++block)
        for(int thread=0;thread<32*c.warps*c.rows;++thread) {
            int lane=thread%32,warp=thread/32,group=warp/c.warps,subwarp=warp%c.warps;
            int row=block*c.rows+group;
            if(lane==0 && subwarp==0 && row<rows) ++outputs[row];
        }
    for(int count:columns) if(count!=1) throw std::runtime_error("column omitted or duplicated");
    for(int count:outputs) if(count!=1) throw std::runtime_error("output omitted or duplicated");
}
int main() {
    int cases=0;
    for(auto c:decode::configs) for(auto s:decode::shapes("correctness"))
        for(int offset:{0,2,4,8}) {
            audit(c,s.intermediate,s.hidden,offset);
            audit(c,s.hidden,s.intermediate,offset);
            cases+=2;
        }
    std::cout<<cases<<" portable launch/address coverage cases passed\n";
}
