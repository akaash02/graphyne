#pragma once

#include <vector>
#include <utility>
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <functional>
#include <cstddef>
#include <random>

struct Node
{
    int id;
    std::vector<float> data;
    std::vector<vector<int>> neighbors; // will have to push_back when adding node for beam search later
    bool deleted = false;
};

class HNSW
{
    int M;
    int efConstruction;
    int maxLayer;
    int entryPointId;
    std::vector<Node> nodes;

    float dist(const vector<float> &a, const vector<float> &b);

    int greedyDescend(const vector<float> &query, int targetLayer);

    std::vector<pair<float, int>> beamSearch(const vector<float> &query, int currentLayer, int ef, int ep);

public:
    HNSW(int M, int efConstruction)
        : M(M),
          efConstruction(efConstruction),
          maxLayer(0),
          entryPointId(-1)
    {
        nodes.reserve(10000); // temporary
    }

    int sampleLayer();
    void addNode(int id, const vector<float> &data);
    void deleteNode(int id);
    std::vector<int> searchANN(const vector<float> &query, int k, int ef);
};
