#pragma once

#include "Tensor.hpp"

class Dataset {
    public:
    virtual ~Dataset() = default;
    virtual size_t size() = 0;

    const virtual Tensor get_item(const size_t idx) const = 0;
    const virtual Tensor get_label(const size_t idx) const = 0;
    
    public:

    const Tensor get_item_batch(size_t batch_size);
    const Tensor get_label_batch(size_t batch_size);
};