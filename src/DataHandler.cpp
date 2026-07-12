#include "DataHandler.hpp"



const Tensor Dataset::get_item_batch(size_t batch_size) {
    std::vector<Tensor> tensors;

    static thread_local std::mt19937 generator(std::random_device{}());
    

    std::uniform_int_distribution<int> distribution(0, size());
    

    for (int i = 0; i < batch_size; i++) {
        tensors.push_back(get_item(distribution(generator)));
    }

    Tensor dst({batch_size, tensors[0].shape[1]});

    Tensor::stack(tensors, dst);

    return dst;
}

const Tensor Dataset::get_label_batch(size_t batch_size) {
    std::vector<Tensor> tensors;

    static thread_local std::mt19937 generator(std::random_device{}());
    

    std::uniform_int_distribution<int> distribution(0, size());
    

    for (int i = 0; i < batch_size; i++) {
        tensors.push_back(get_label(distribution(generator)));
    }

    Tensor dst({batch_size, tensors[0].shape[1]});

    Tensor::stack(tensors, dst);

    return dst;
}