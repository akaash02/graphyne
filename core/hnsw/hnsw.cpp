#include "hnsw.hpp"

using namespace std;

struct Node
{
    int id;
    vector<float> data;
    vector<vector<int>> neighbors;
    bool deleted = false;
};

class HNSW
{
    int M;
    int efConstruction;
    int maxLayer;
    Node entryPoint;
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
        int currentNodeId = entryPoint.id;
        int currentLayer = maxLayer;
        float bestDist = this->dist(query, entryPoint.data);

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

public:
    HNSW(int M, int efConstruction);

    void addNode(int id, const vector<float> &data);
    void deleteNode(int id);
    vector<int> searchANN(const vector<float> &query, int k, int ef);
};