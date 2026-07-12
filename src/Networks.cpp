#include "Networks.hpp"

Tensor Sequential::forward(Tensor& x){
    activations.clear();
    activations.push_back(x);
    Tensor new_activation = x;

    for (const Layer* current_layer: layers) {
        new_activation = current_layer->forward(new_activation);

        activations.push_back(new_activation);
    }

    return new_activation;
}

void Sequential::backward(Tensor loss, float lr_rate) {
    Tensor current_loss = loss;

    for (int i = layers.size() - 1; i >= 0; i--) {
        
        // 1. APPLY RELU MASK
        size_t rows = current_loss.shape[0];
        size_t cols = current_loss.shape[1];

        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                float activation_value = *activations[i + 1].at({r, c});

                *current_loss.at({r, c}) *= activation_value * (1.0f - activation_value);
            }
        }

        // 2. WEIGHT GRADIENTS (dw = current_loss * activations[i]^T)
        Tensor prev_act_T = activations[i];
        prev_act_T.transpose();
        Tensor dw({prev_act_T.shape[0], current_loss.shape[1]});
        Tensor::matmul(prev_act_T, current_loss, dw);

        // 3. BIAS GRADIENTS
        Tensor db = current_loss;

        // Apply learning rate
        dw.multiply(lr_rate);
        db.multiply(lr_rate);
        
        // Update layer values
        layers[i]->weights.subtract(dw);
        layers[i]->biases.subtract(db);

        // 4. PREPARE LOSS FOR THE PREVIOUS LAYER (next_loss = weights^T * current_loss)
        Tensor weights_T = layers[i]->weights;
        weights_T.transpose();

        Tensor next_loss({current_loss.shape[0], weights_T.shape[1]});
        Tensor::matmul(current_loss, weights_T, next_loss);

        current_loss = next_loss;
    }
}
