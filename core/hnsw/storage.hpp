#pragma once

#include <vector>
#include <cstddef>
#include <malloc.h>
#include <new>


class NodeBuffer{
    float* data_ = nullptr;
    size_t n_nodes = 0;
    size_t dim_ = 0;

public:
    NodeBuffer() = default;
    
    NodeBuffer(size_t n_nodes, size_t dim) : n_nodes(n_nodes), dim_(dim) {
        size_t bytes = ((n_nodes * dim_ * sizeof(float) + 31) / 32) * 32;
        data_ = static_cast<float*>(_aligned_malloc(bytes, 32));
        if (!data_) throw std::bad_alloc{};
    }

    ~NodeBuffer() { _aligned_free(data_); }

    // forbidden copy constructor and coppy assign
    NodeBuffer(const NodeBuffer&) = delete;
    NodeBuffer& operator=(const NodeBuffer&) = delete;

    // move constructor
    NodeBuffer(NodeBuffer&& other) noexcept {
        data_ = other.data_;
        n_nodes = other.n_nodes;
        dim_  = other.dim_;
        other.data_ = nullptr;
        other.n_nodes = 0;
        other.dim_  = 0;
    }

    // move assign
    NodeBuffer& operator=(NodeBuffer&& other) noexcept {
        if (this != &other) { 
            _aligned_free(data_);

            data_ = other.data_;
            n_nodes = other.n_nodes;
            dim_  = other.dim_;

            other.data_ = nullptr;
            other.n_nodes = 0;
            other.dim_  = 0;
        }
        return *this;
    }


    float& operator[](size_t i) 
    {
        return data_[i];
    }

    const float& operator[](size_t i) const
    {
        return data_[i];
    }
    
    float* row(size_t i) 
    {
        return data_ + i * dim_;
    }
    
    const float* row(size_t i) const 
    {
        return data_ + i * dim_;
    }
    
    float* data() 
    { 
        return data_;
    }

    const float* data() const
    { 
        return data_;
    }
    
    size_t dim() const 
    { 
        return dim_; 
    }

    // returns full size
    size_t size() const
    {
        return n_nodes * dim_;
    }
};