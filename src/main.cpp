#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include "Tensor.hpp"
#include "Layers.hpp"
#include "Networks.hpp"
#include "LossF.hpp"
#include "DataHandler.hpp"

/*
class MNISTDataset: public Dataset {

    std::vector<uint8_t> images;
    std::vector<uint8_t> labels;

    size_t num_samples;
    size_t image_size = 28 * 28;

    std::vector<uint8_t> initalize_images(std::string images_path) {
        std::ifstream images_file(images_path, std::ios::binary | std::ios::ate);

        if (!images_file.is_open()) {
            std::cerr << "Error: Could not open file " << images_path << std::endl;
            return {}; // Return empty vector if file missing
        }

        size_t file_size = images_file.tellg();
        images_file.seekg(16, std::ios::beg);

        file_size -= 16;

        std::vector<uint8_t> data_buffer(file_size);

        images_file.read(reinterpret_cast<char*>(data_buffer.data()), file_size);

        return data_buffer;
    }

    std::vector<uint8_t> initialize_labels(const std::string& labels_path) {
        std::ifstream labels_file(labels_path, std::ios::binary | std::ios::ate);

        if (!labels_file.is_open()) {
            std::cerr << "Error: Could not open file " << labels_path << std::endl;
            return {}; 
        }

        size_t file_size = labels_file.tellg();

        labels_file.seekg(8, std::ios::beg);

        size_t data_size = file_size - 8;
        std::vector<uint8_t> data_buffer(data_size);

        labels_file.read(reinterpret_cast<char*>(data_buffer.data()), data_size);

        return data_buffer;
    }

    
    public:
    MNISTDataset(std::string images_path, std::string labels_path) {
        images = initalize_images(images_path);
        labels = initialize_labels(labels_path);
        num_samples = images.size() / image_size;
    };

    size_t size() override{
        return num_samples;
    }

    const Tensor get_item(size_t idx) const override{
        size_t start_index = idx * image_size;

        Tensor image({1, image_size}); 

        for (size_t i = 0; i <  image_size; i++) {
            *image.at({0, i}) = images[i + start_index] / 255.0f;
        }

        return image;
    }

    const Tensor get_label(size_t idx) const override {
        Tensor label({1, 10});
        *label.at({0, static_cast<size_t>(labels[idx]-1)}) = 1;
        return label;
    }

};*/

void test_model(std::vector<Tensor>& xor_inputs, std::vector<Tensor>& xor_targets, Sequential& model) {
    std::cout << "\n=== Testing model ===" << std::endl;
    for (size_t i = 0; i < xor_inputs.size(); ++i) {
        Tensor out = model.forward(xor_inputs[i]);
        std::cout << "Input index " << i << " | Predicted Output: ";
        out.print_tensor(); 
    }
}


int main() {
    srand(time(0));

    std::vector<Layer*> network_layers = {
        new DenseLayer(2, 2), // Hidden layer: 2 inputs, 4 neurons 1 output
        new DenseLayer(2, 1),
    };

    network_layers[1]->use_relu = true;
    
    Sequential model(network_layers);
    MSELoss loss_fn;
    std::vector<Tensor> xor_inputs = {
        Tensor({0.0f, 0.0f}, {1, 2}),
        Tensor({0.0f, 1.0f}, {1, 2}),
        Tensor({1.0f, 0.0f}, {1, 2}),
        Tensor({1.0f, 1.0f}, {1, 2})
    };

    std::cout << xor_inputs.size() << std::endl;

    std::vector<Tensor> xor_targets = {
        Tensor({0.0f}, {1}),
        Tensor({1.0f}, {1}),
        Tensor({1.0f}, {1}),
        Tensor({0.0f}, {1})
    };

    float lr_rate = 1e-1;
    int epochs = 10000; 
    size_t batch_size = 1;

    test_model(xor_inputs, xor_targets, model);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_epoch_loss = 0.0f;

        for (size_t i = 0; i < (xor_inputs.size() / batch_size); ++i) {
            //std::cout << "item" << std::endl;
            Tensor x = xor_inputs[i];
            Tensor y = xor_targets[i];

            Tensor prediction = model.forward(x);

            float loss = loss_fn.compute(prediction, y);

            total_epoch_loss += loss;

            Tensor loss_grad({batch_size, prediction.shape[1]});

            loss_fn.gradient(prediction, y, loss_grad);
        
            model.backward(loss_grad, lr_rate);
        }

        // Print progress every 50 epochs
        if ((epoch + 1) % 1000 == 0) {
            std::cout << "Epoch " << (epoch + 1) << " | Avg Loss: " << (total_epoch_loss / 4.0f) << std::endl;
        }
    }

    test_model(xor_inputs, xor_targets, model);
    
    return 0;
}
