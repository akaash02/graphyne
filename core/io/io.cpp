#include "io.hpp"
#include <fstream>
#include <stdexcept>

std::ifstream open_and_read_header(const std::string &path, int32_t &rows, int32_t &cols)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open " + path);

    f.read(reinterpret_cast<char *>(&rows), sizeof rows);
    f.read(reinterpret_cast<char *>(&cols), sizeof cols);
    if (!f)
        throw std::runtime_error("cannot read header in " + path);
    if (rows <= 0 || cols <= 0)
        throw std::runtime_error("bad header in " + path);

    return f;
}

void read_payload(std::ifstream &f, void *dest, size_t bytes, const std::string &path)
{
    f.read(reinterpret_cast<char *>(dest), static_cast<std::streamsize>(bytes));
    if (static_cast<size_t>(f.gcount()) != bytes)
        throw std::runtime_error("truncated read in " + path);
}

std::pair<int32_t, int32_t> read_header(const std::string &path)
{
    int32_t rows = 0, cols = 0;
    open_and_read_header(path, rows, cols);
    return {rows, cols};
}

NodeBuffer load_vectors(const std::string &path)
{
    int32_t rows = 0, cols = 0;
    std::ifstream f = open_and_read_header(path, rows, cols);

    NodeBuffer buf(rows, cols);
    read_payload(f, buf.data(), size_t(rows) * size_t(cols) * sizeof(float), path);
    return buf;
}

void load_vectors_into(const std::string &path, float *dest, int32_t expected_rows, int32_t expected_cols)
{
    int32_t rows = 0, cols = 0;
    std::ifstream f = open_and_read_header(path, rows, cols);

    if (rows != expected_rows || cols != expected_cols)
        throw std::runtime_error(
            "dimension mismatch in " + path +
            ": file is " + std::to_string(rows) + "x" + std::to_string(cols) +
            ", caller expected " + std::to_string(expected_rows) + "x" + std::to_string(expected_cols));

    read_payload(f, dest, size_t(rows) * size_t(cols) * sizeof(float), path);
}

std::vector<int32_t> load_ids(const std::string &path, int32_t &out_rows, int32_t &out_cols)
{
    std::ifstream f = open_and_read_header(path, out_rows, out_cols);

    std::vector<int32_t> out(size_t(out_rows) * size_t(out_cols));
    read_payload(f, out.data(), out.size() * sizeof(int32_t), path);
    return out;
}
