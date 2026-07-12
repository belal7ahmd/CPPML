#pragma once

#include "Tensor.hpp"

class Layer {
public:
    Layer() = default;

    virtual ~Layer() = default;

    Tensor weights;
    Tensor biases;

    bool use_relu;
    
    // Virtual functions that every layer type must implement
    virtual Tensor forward(const Tensor& x) const = 0;
};

class DenseLayer : public Layer {
public:
    
    DenseLayer(size_t input_dim, size_t output_dim);

    Tensor forward(const Tensor& x) const override;
};


