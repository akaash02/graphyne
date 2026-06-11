#include "hnsw.hpp"

using namespace std;

struct Node
{
    int id;
    vector<float> data;
    vector<vector<int>> neighbors; // will have to push_back when adding node for beam search later
    bool deleted = false;
};

class HNSW
{
    int M;
    int efConstruction;
    int maxLayer;
    int entryPointId;
    vector<Node> nodes;

    float dist(const vector<float> &a, const vector<float> &b)
    {
        float sum = 0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
        }

        return sum;
    }

    int greedyDescend(const vector<float> &query, int targetLayer)
    {
        int currentNodeId = entryPointId;
        int currentLayer = maxLayer;
        float bestDist = this->dist(query, nodes[entryPointId].data);

        while (currentLayer > targetLayer)
        {
            bool changed = false;

            for (int neighborId : nodes[currentNodeId].neighbors[currentLayer])
            {
                float d = this->dist(query, nodes[neighborId].data);

                if (d < bestDist)
                {
                    bestDist = d;
                    currentNodeId = neighborId;
                    changed = true;
                }
            }

            if (!changed)
            {
                currentLayer--;
            }
        }

        return currentNodeId;
    }

    vector<pair<float, int>> beamSearch(const vector<float> &query, int currentLayer, int ef, int ep)
    {
        priority_queue<pair<float, int>, vector<pair<float, int>>, greater<pair<float, int>>> candidates;
        priority_queue<pair<float, int>> results;
        unordered_set<int> visited;

        float epDist = dist(query, nodes[ep].data);
        candidates.push({epDist, ep});
        results.push({epDist, ep});
        visited.insert(ep);

        while (!candidates.empty())
        {
            pair<float, int> best = candidates.top();
            float bestCandDist = best.first;
            int bestCandId = best.second;

            candidates.pop();

            if (bestCandDist > results.top().first)
                break;

            for (int neighborId : nodes[bestCandId].neighbors[currentLayer])
            {
                if (visited.find(neighborId) == visited.end())
                {
                    visited.insert(neighborId);
                    float d = dist(query, nodes[neighborId].data);

                    if (results.size() < ef || d < results.top().first)
                    {
                        candidates.push({d, neighborId});
                        results.push({d, neighborId});

                        if (results.size() > ef)
                        {
                            results.pop();
                        }
                    }
                }
            }
        }

        vector<pair<float, int>> out;
        out.reserve(results.size());
        while (!results.empty())
        {
            out.push_back(results.top());
            results.pop();
        }
        reverse(out.begin(), out.end());
        return out;
    }

public:
    HNSW(int M, int efConstruction)
        : M(M),
          efConstruction(efConstruction),
          maxLayer(0),
          entryPointId(-1)
    {
        nodes.reserve(10000); // temporary
    }

    void addNode(int id, const vector<float> &data);
    void deleteNode(int id);
    vector<int> searchANN(const vector<float> &query, int k, int ef);
};