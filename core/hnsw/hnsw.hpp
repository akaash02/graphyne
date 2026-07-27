#pragma once

#include <vector>
#include <utility>
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <functional>
#include <cstddef>
#include <cmath>
#include <random>
#include <limits>
#include "storage.hpp"
#include "neighbors.hpp"

class HNSW
{
    int M;
    int efConstruction;
    int maxLayer;
    int maxLayersCap;
    int entryPointId;
    std::mt19937 rng;

    NodeBuffer    nodeBuffer;
    NeighborStore neighborStore;

    std::vector<int>     nodeLayer_; // assigned layer per node
    std::vector<uint8_t> deleted_;   // soft-delete flag per node

    int sampleLayer();

    float dist(const float* a, const float* b);

    int greedyDescend(const float* query, int targetLayer);

    std::vector<std::pair<float, int>> beamSearch(const float* query, int currentLayer, int ef, int ep);

    std::vector<std::pair<float, int>> selectNeighbors(const float* query, std::vector<std::pair<float, int>> candidates, int maxConnections);

    void wireNeighbors(int newId, std::vector<std::pair<float, int>> &selected, int currentLayer);

public:
    HNSW(int M, int efConstruction, size_t n_nodes, size_t dim)
        : M(M),
          efConstruction(efConstruction),
          maxLayer(0),
          maxLayersCap((int)(std::log((double)n_nodes) / std::log((double)M)) + 2),
          entryPointId(-1),
          rng(std::random_device{}()),
          nodeBuffer(n_nodes, dim),
          neighborStore(n_nodes, M, maxLayersCap - 1),
          nodeLayer_(n_nodes, 0),
          deleted_(n_nodes, 0)
    {}

    void addNode(int newId);
    void deleteNode(int id);
    std::vector<int> searchANN(const std::vector<float> &query, int k, int ef);
};
