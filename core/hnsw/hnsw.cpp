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
    int entryPoint;
    vector<Node> nodes;

public:
    HNSW(int M, int efConstruction);

    void addNode(int id, const vector<float> &data);
    void deleteNode(int id);
    vector<int> searchANN(const vector<float> &query, int k, int ef);
};