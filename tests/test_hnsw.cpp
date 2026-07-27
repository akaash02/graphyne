#include "../io/io.hpp"
#include <cstdio>

int main(){
    NodeBuffer nodeBuffer = load_matrix("bin/train.bin");
    std::printf("loaded %zu, dim %zu\n", (nodeBuffer.size() / nodeBuffer.dim()), nodeBuffer.dim());

    return 0;
}