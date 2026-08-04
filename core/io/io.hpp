#pragma once
#include "../hnsw/storage.hpp"
#include <string>
#include <vector>
#include <cstdint>
#include <utility>

std::pair<int32_t, int32_t> read_header(const std::string &path);

NodeBuffer load_vectors(const std::string &path);

void load_vectors_into(const std::string &path, float *dest, int32_t expected_rows, int32_t expected_cols);

std::vector<int32_t> load_ids(const std::string &path, int32_t &out_rows, int32_t &out_cols);
