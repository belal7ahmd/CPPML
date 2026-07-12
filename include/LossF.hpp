#pragma once

#include "Tensor.hpp"

class MSELoss {
    public:
    static void gradient(const Tensor& pred, const Tensor& label, Tensor& loss);
    static float compute(const Tensor& pred, const Tensor& label);
};
