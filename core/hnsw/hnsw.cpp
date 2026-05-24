#include "hnsw.hpp"

using namespace std;

struct Node
{
    int id;
    vector<float> data;
    vector<std::vector<int>> neighbors;
    bool deleted = false;
};