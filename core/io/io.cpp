#include "io.hpp"
#include <fstream>
#include <stdexcept>
#include <cstdint>


NodeBuffer load_matrix(const std::string& path){
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);

    int32_t rows = 0, cols = 0;
    f.read(reinterpret_cast<char*>(&rows), sizeof rows);
    f.read(reinterpret_cast<char*>(&cols), sizeof cols);

    NodeBuffer buf(rows, cols);
    size_t expected = size_t(rows) * cols * sizeof(float);
    f.read(reinterpret_cast<char*>(buf.data()), expected);
    if (static_cast<size_t>(f.gcount()) != expected)
        throw std::runtime_error("truncated read in " + path);

    return buf;
}