#pragma once

#include <cstddef>
#include <cstring>
#include <cstdint>
#include <cassert>
#include <malloc.h>
#include <new>

class NeighborStore
{
    int      *data_   = nullptr; // [nodeId * (maxLayers+1) * 2M + layer * 2M + slot]
    uint16_t *counts_ = nullptr; // [nodeId * (maxLayers+1) + layer]
    size_t N_         = 0;
    size_t M2_        = 0; // 2 * M — uniform row width for every layer
    int    maxLayers_ = 0;

    size_t rowStride()  const { return M2_; }
    size_t nodeStride() const { return (maxLayers_ + 1) * M2_; }

public:
    NeighborStore() = default;

    NeighborStore(size_t N, size_t M, int maxLayers)
        : N_(N), M2_(2 * M), maxLayers_(maxLayers)
    {
        size_t dataBytes  = ((N_ * (maxLayers_ + 1) * M2_ * sizeof(int) + 31) / 32) * 32;
        size_t countBytes = ((N_ * (maxLayers_ + 1) * sizeof(uint16_t) + 31) / 32) * 32;

        data_   = static_cast<int *>(_aligned_malloc(dataBytes, 32));
        counts_ = static_cast<uint16_t *>(_aligned_malloc(countBytes, 32));
        if (!data_ || !counts_) throw std::bad_alloc{};

        std::memset(counts_, 0, countBytes);
    }

    ~NeighborStore()
    {
        _aligned_free(data_);
        _aligned_free(counts_);
    }

    NeighborStore(const NeighborStore &) = delete;
    NeighborStore &operator=(const NeighborStore &) = delete;

    NeighborStore(NeighborStore &&o) noexcept
        : data_(o.data_), counts_(o.counts_), N_(o.N_), M2_(o.M2_), maxLayers_(o.maxLayers_)
    {
        o.data_      = nullptr;
        o.counts_    = nullptr;
        o.N_         = 0;
        o.M2_        = 0;
        o.maxLayers_ = 0;
    }

    NeighborStore &operator=(NeighborStore &&o) noexcept
    {
        if (this != &o)
        {
            _aligned_free(data_);
            _aligned_free(counts_);
            data_      = o.data_;      o.data_      = nullptr;
            counts_    = o.counts_;    o.counts_    = nullptr;
            N_         = o.N_;         o.N_         = 0;
            M2_        = o.M2_;        o.M2_        = 0;
            maxLayers_ = o.maxLayers_; o.maxLayers_ = 0;
        }
        return *this;
    }

    int *neighbors(int nodeId, int layer)
    {
        return data_ + nodeId * nodeStride() + layer * rowStride();
    }

    const int *neighbors(int nodeId, int layer) const
    {
        return data_ + nodeId * nodeStride() + layer * rowStride();
    }

    int count(int nodeId, int layer) const
    {
        return counts_[nodeId * (maxLayers_ + 1) + layer];
    }

    int capacity() const { return static_cast<int>(M2_); }

    void addNeighbor(int nodeId, int layer, int neighborId)
    {
        uint16_t &c = counts_[nodeId * (maxLayers_ + 1) + layer];
        assert(c < M2_);
        neighbors(nodeId, layer)[c] = neighborId;
        ++c;
    }

    void setNeighbors(int nodeId, int layer, const int *ids, int n)
    {
        std::memcpy(neighbors(nodeId, layer), ids, n * sizeof(int));
        counts_[nodeId * (maxLayers_ + 1) + layer] = static_cast<uint16_t>(n);
    }
};
