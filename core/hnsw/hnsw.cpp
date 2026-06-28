#include "hnsw.hpp"

using namespace std;

int HNSW::sampleLayer()
{
    uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    if (r <= 0.0f)
        r = std::numeric_limits<float>::min(); // avoid -log(0) = inf
    return (int)floor(-log(r) * (1.0f / log(M)));
}

float HNSW::dist(const vector<float> &a, const vector<float> &b)
{
    float sum = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }

    return sum;
}

// Diversity heuristic (Malkov & Yashunin, Algorithm 4): pick up to maxConnections
// neighbors, accepting a candidate only if it is closer to the query than to any
// already-selected neighbor. Spreads edges across space instead of clustering.
vector<pair<float, int>> HNSW::selectNeighbors(const vector<float> &query, vector<pair<float, int>> candidates, int maxConnections)
{
    sort(candidates.begin(), candidates.end()); // nearest first

    vector<pair<float, int>> selected;
    selected.reserve(maxConnections);

    for (const auto &cand : candidates)
    {
        if ((int)selected.size() >= maxConnections)
            break;

        float distToQuery = cand.first;
        bool keep = true;

        for (const auto &s : selected)
        {
            float distToSelected = dist(nodes[cand.second].data, nodes[s.second].data);
            if (distToSelected < distToQuery)
            {
                keep = false; // closer to an existing neighbor than to the query
                break;
            }
        }

        if (keep)
            selected.push_back(cand);
    }

    return selected;
}

// Wire the already-inserted node (newId) to each selected neighbor bidirectionally,
// then prune any neighbor that now exceeds its connection cap using the heuristic.
// Assumes nodes[newId] already exists in `nodes`, so no resize happens here.
void HNSW::wireNeighbors(int newId, vector<pair<float, int>> &selected, int currentLayer)
{
    int maxConnections = (currentLayer == 0) ? 2 * M : M;

    for (const auto &[d, neighborId] : selected)
    {
        nodes[newId].neighbors[currentLayer].push_back(neighborId);
        nodes[neighborId].neighbors[currentLayer].push_back(newId);

        auto &conn = nodes[neighborId].neighbors[currentLayer];
        if ((int)conn.size() > maxConnections)
        {
            vector<pair<float, int>> candidates;
            candidates.reserve(conn.size());
            for (int cid : conn)
            {
                candidates.push_back({dist(nodes[neighborId].data, nodes[cid].data), cid});
            }

            vector<pair<float, int>> kept = selectNeighbors(nodes[neighborId].data, candidates, maxConnections);

            conn.clear();
            for (const auto &[kd, kid] : kept)
            {
                conn.push_back(kid);
            }
        }
    }
}

int HNSW::greedyDescend(const vector<float> &query, int targetLayer)
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

vector<pair<float, int>> HNSW::beamSearch(const vector<float> &query, int currentLayer, int ef, int ep)
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

void HNSW::addNode(const vector<float> &data)
{
    int newId = (int)nodes.size();
    int assignedLayer = sampleLayer();

    Node node{.id = newId, .data = data};
    node.neighbors.resize(assignedLayer + 1);
    // Insert the node up front so wiring can mutate its lists in place via nodes[newId].
    // nodes.reserve(10000) keeps this push_back from reallocating during typical builds.
    nodes.push_back(std::move(node));

    // First node bootstraps the index: it becomes the entry point with no edges.
    if (entryPointId == -1)
    {
        entryPointId = newId;
        maxLayer = assignedLayer;
        return;
    }

    int ep = entryPointId;

    // Phase A: greedily descend the upper layers (ef=1, no wiring) down to the
    // node's assigned layer, carrying the closest node found as the entry point.
    if (assignedLayer < maxLayer)
    {
        ep = greedyDescend(data, assignedLayer);
    }

    // Phase B: from the highest layer the node shares with the graph down to layer 0,
    // beam-search, pick diverse neighbors, and wire them bidirectionally.
    int startLayer = std::min(assignedLayer, maxLayer);
    for (int layer = startLayer; layer >= 0; layer--)
    {
        vector<pair<float, int>> candidates = beamSearch(data, layer, efConstruction, ep);

        int maxConnections = (layer == 0) ? 2 * M : M;
        vector<pair<float, int>> selected = selectNeighbors(data, candidates, maxConnections);

        wireNeighbors(newId, selected, layer);

        ep = candidates[0].second; // closest result seeds the next layer down
    }

    // If the node reached above the current top, it becomes the new entry point.
    if (assignedLayer > maxLayer)
    {
        entryPointId = newId;
        maxLayer = assignedLayer;
    }
}
