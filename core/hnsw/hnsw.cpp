#include "hnsw.hpp"
#include <cassert>

using namespace std;

int HNSW::sampleLayer()
{
    uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    if (r <= 0.0f)
        r = std::numeric_limits<float>::min();
    int layer = (int)floor(-log(r) * (1.0f / log(M)));
    return min(layer, maxLayersCap - 1);
}

float HNSW::dist(const float* a, const float* b)
{
    float sum = 0;
    for (size_t i = 0; i < nodeBuffer.dim(); ++i)
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    return sum;
}

vector<pair<float, int>> HNSW::selectNeighbors(const float* query, vector<pair<float, int>> candidates, int maxConnections)
{
    sort(candidates.begin(), candidates.end());

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
            float distToSelected = dist(nodeBuffer.row(cand.second), nodeBuffer.row(s.second));
            if (distToSelected < distToQuery)
            {
                keep = false;
                break;
            }
        }

        if (keep)
            selected.push_back(cand);
    }

    return selected;
}

void HNSW::wireNeighbors(int newId, vector<pair<float, int>> &selected, int currentLayer)
{
    int maxConnections = (currentLayer == 0) ? 2 * M : M;

    // newId gets one unconditional append per selected neighbor here, then reciprocal
    // appends from future insertions go through the >= guard below. This is safe because
    // every row has 2*M capacity regardless of layer. If upper-layer rows are ever
    // tightened to M, the unchecked append will overflow.
    for (const auto &[d, neighborId] : selected)
    {
        neighborStore.addNeighbor(newId, currentLayer, neighborId);

        if (neighborStore.count(neighborId, currentLayer) >= maxConnections)
        {
            int        cnt = neighborStore.count(neighborId, currentLayer);
            const int *row = neighborStore.neighbors(neighborId, currentLayer);

            vector<pair<float, int>> candidates;
            candidates.reserve(cnt + 1);
            for (int i = 0; i < cnt; ++i)
                candidates.push_back({dist(nodeBuffer.row(neighborId), nodeBuffer.row(row[i])), row[i]});
            candidates.push_back({d, newId});

            vector<pair<float, int>> kept = selectNeighbors(nodeBuffer.row(neighborId), std::move(candidates), maxConnections);

            vector<int> keptIds;
            keptIds.reserve(kept.size());
            for (const auto &[kd, kid] : kept)
                keptIds.push_back(kid);

            neighborStore.setNeighbors(neighborId, currentLayer, keptIds.data(), (int)keptIds.size());
        }
        else
        {
            neighborStore.addNeighbor(neighborId, currentLayer, newId);
        }
    }
}

int HNSW::greedyDescend(const float* query, int targetLayer)
{
    int currentNodeId = entryPointId;
    int currentLayer  = maxLayer;
    float bestDist    = this->dist(query, nodeBuffer.row(entryPointId));

    while (currentLayer > targetLayer)
    {
        bool changed = true;
        while (changed)
        {
            changed = false;
            int bestId = currentNodeId;

            int        cnt = neighborStore.count(currentNodeId, currentLayer);
            const int *row = neighborStore.neighbors(currentNodeId, currentLayer);
            for (int i = 0; i < cnt; ++i)
            {
                float d = this->dist(query, nodeBuffer.row(row[i]));
                if (d < bestDist)
                {
                    bestDist = d;
                    bestId   = row[i];
                    changed  = true;
                }
            }

            currentNodeId = bestId;
        }
        currentLayer--;
    }

    return currentNodeId;
}

vector<pair<float, int>> HNSW::beamSearch(const float* query, int currentLayer, int ef, int ep)
{
    priority_queue<pair<float, int>, vector<pair<float, int>>, greater<pair<float, int>>> candidates;
    priority_queue<pair<float, int>> results;
    unordered_set<int> visited;

    float epDist = dist(query, nodeBuffer.row(ep));
    candidates.push({epDist, ep});
    results.push({epDist, ep});
    visited.insert(ep);

    while (!candidates.empty())
    {
        auto [bestCandDist, bestCandId] = candidates.top();
        candidates.pop();

        if (bestCandDist > results.top().first)
            break;

        int        cnt = neighborStore.count(bestCandId, currentLayer);
        const int *row = neighborStore.neighbors(bestCandId, currentLayer);
        for (int i = 0; i < cnt; ++i)
        {
            int neighborId = row[i];
            if (visited.find(neighborId) == visited.end())
            {
                visited.insert(neighborId);
                float d = dist(query, nodeBuffer.row(neighborId));

                if ((int)results.size() < ef || d < results.top().first)
                {
                    candidates.push({d, neighborId});
                    results.push({d, neighborId});

                    if ((int)results.size() > ef)
                        results.pop();
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

void HNSW::addNode(int newId)
{
    int assignedLayer = sampleLayer();
    nodeLayer_[newId] = assignedLayer;

    if (entryPointId == -1)
    {
        entryPointId = newId;
        maxLayer     = assignedLayer;
        return;
    }

    int ep = entryPointId;

    if (assignedLayer < maxLayer)
        ep = greedyDescend(nodeBuffer.row(newId), assignedLayer);

    int startLayer = min(assignedLayer, maxLayer);
    for (int layer = startLayer; layer >= 0; layer--)
    {
        vector<pair<float, int>> candidates = beamSearch(nodeBuffer.row(newId), layer, efConstruction, ep);
        assert(!candidates.empty());
        ep = candidates[0].second;

        int maxConnections = (layer == 0) ? 2 * M : M;
        vector<pair<float, int>> selected = selectNeighbors(nodeBuffer.row(newId), std::move(candidates), maxConnections);

        wireNeighbors(newId, selected, layer);
    }

    if (assignedLayer > maxLayer)
    {
        entryPointId = newId;
        maxLayer     = assignedLayer;
    }
}

void HNSW::deleteNode(int id)
{
    assert(entryPointId == -1 || nodeLayer_[entryPointId] == maxLayer);

    deleted_[id] = 1;

    if (id != entryPointId)
        return;

    // NOTE: the promoted node may sit at a lower layer than other live nodes, leaving
    // upper layers unreachable until a compaction/rebuild pass corrects maxLayer.

    // Find a non-deleted neighbor to promote, preferring higher layers.
    int bestCandidate = -1;
    int bestLayer     = -1;

    for (int layer = nodeLayer_[id]; layer >= 0; --layer)
    {
        int        cnt = neighborStore.count(id, layer);
        const int *row = neighborStore.neighbors(id, layer);
        for (int i = 0; i < cnt; ++i)
        {
            int candidate = row[i];
            if (!deleted_[candidate] && nodeLayer_[candidate] > bestLayer)
            {
                bestCandidate = candidate;
                bestLayer     = nodeLayer_[candidate];
            }
        }
    }

    if (bestCandidate == -1)
    {
        entryPointId = -1;
        maxLayer     = 0;
    }
    else
    {
        entryPointId = bestCandidate;
        maxLayer     = bestLayer;
    }
}

vector<int> HNSW::searchANN(const vector<float> &query, int k, int ef)
{
    if (entryPointId == -1)
        return {};

    int ep = greedyDescend(query.data(), 1);

    vector<pair<float, int>> candidates = beamSearch(query.data(), 0, ef, ep);

    vector<int> results;
    results.reserve(k);
    for (const auto &[d, id] : candidates)
    {
        if (deleted_[id])
            continue;
        results.push_back(id);
        if ((int)results.size() == k)
            break;
    }
    return results;
}
