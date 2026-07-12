#pragma once

#include "Layers.hpp"
#include <variant>
#include <vector>

class Sequential {
    public:
    std::vector<Layer*> layers;
    std::vector<Tensor> activations;

    Sequential(std::vector<Layer*> layers): layers(layers) {};
    ~Sequential() {
        for (Layer* layer: layers) {
            delete layer;
        }
    };

    Tensor forward(Tensor& x);

    void backward(Tensor loss, float lr_rate);
};