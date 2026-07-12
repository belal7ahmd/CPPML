#include "Layers.hpp"

// In place relu
void ReLU(Tensor& x) {
    for (float& val : x) {
        val = 1.0 / (1.0 + std::exp(-val));
    }
}

DenseLayer::DenseLayer(size_t input_dim, size_t output_dim): 
    Layer()
{
    // Initialize your network parameters
    weights = Tensor({input_dim, output_dim});
    biases  = Tensor({1, output_dim});
    use_relu = true;

    weights.rand_init(-1.0f, 1.0f);
    biases.rand_init(-1.0f, 1.0f);
}

Tensor DenseLayer::forward(const Tensor& x) const {
    Tensor new_activations = Tensor({x.shape[0], weights.shape[1]});

    /*x.print_tensor();
    weights.print_tensor();*/
    Tensor::matmul(x, weights, new_activations);
    

    size_t batch_size = new_activations.shape[0];
    size_t output_dim = new_activations.shape[1];


    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < output_dim; ++j) {

            *new_activations.at({i, j}) += *biases.at({0, j});
        }
    }
    if (use_relu) {
        ReLU(new_activations);
    }
  

    return new_activations;
}
