#include "LossF.hpp"


void MSELoss::gradient(const Tensor& pred, const Tensor& label, Tensor& loss) {
    for (size_t i = 0; i < pred.shape[0]; i++) {
        for (size_t j = 0; j < pred.shape[1]; j++) {
            // Gradient = Prediction - Target
            *loss.at({i, j}) = *pred.at({i, j}) - *label.at({i, j}); 
        }
    }
};

float MSELoss::compute(const Tensor& pred, const Tensor& label) {
    float val;
    float loss = 0.0f;
    for (size_t i = 0; i < pred.shape[0]; i++) {
        for (size_t j = 0; j < pred.shape[1]; j++) {

            val = *label.at({i, j}) - *pred.at({i, j});
            loss +=  val * val;
        }
    }
    return loss;
};

