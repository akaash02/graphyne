#pragma once

#include <vector>
#include <utility>
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <functional>
#include <cstddef>
#include <random>
#include <limits>

struct Node
{
    int id;
    std::vector<float> data;
    std::vector<std::vector<int>> neighbors; // will have to push_back when adding node for beam search later
    bool deleted = false;
};

class HNSW
{
    int M;
    int efConstruction;
    int maxLayer;
    int entryPointId;
    std::vector<Node> nodes;
    std::mt19937 rng;

    int sampleLayer();

    float dist(const std::vector<float> &a, const std::vector<float> &b);

    int greedyDescend(const std::vector<float> &query, int targetLayer);

    std::vector<std::pair<float, int>> beamSearch(const std::vector<float> &query, int currentLayer, int ef, int ep);

    std::vector<std::pair<float, int>> selectNeighbors(const std::vector<float> &query, std::vector<std::pair<float, int>> candidates, int maxConnections);

    void wireNeighbors(int newId, std::vector<std::pair<float, int>> &selected, int currentLayer);

public:
    HNSW(int M, int efConstruction)
        : M(M),
          efConstruction(efConstruction),
          maxLayer(0),
          entryPointId(-1),
          rng(std::random_device{}())
    {
        nodes.reserve(10000); // temporary
    }

    void addNode(const std::vector<float> &data);
    void deleteNode(int id);
    std::vector<int> searchANN(const std::vector<float> &query, int k, int ef);
};
