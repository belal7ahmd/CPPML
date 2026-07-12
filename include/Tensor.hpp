#pragma once

#include <iostream>
#include <vector>
#include <cstdlib>
#include  <random>

class Tensor {
    public:
    float* data;

    std::vector<size_t> strides;

    bool is_owner;

    public:

    std::vector<size_t> shape;
    size_t size;

    // Dummy default constructor
    Tensor();
    
    Tensor(std::vector<size_t> tensor_shape);
    Tensor(std::initializer_list<float> values, std::vector<size_t> tensor_shape);


    ~Tensor();

    Tensor(const Tensor& other);

    Tensor& operator=(const Tensor& source);

    static void matmul(const Tensor& mat_a, const Tensor& mat_b, Tensor& dst);
    static void sum(const Tensor& mat_a, const Tensor& mat_b, Tensor& dst);
    static void stack(const std::vector<Tensor>& tensors, Tensor& dst);
    void subtract(const Tensor& tensor_b);
    void multiply(const float scalar);
    void rand_init(float min_range, float max_range);
    void print_tensor() const;
    void transpose();
    void fill(float value);

    // For modifying data: tensor.at({1, 2}) = 5.0f;
    float* at(const std::vector<size_t>& indices);

    // For read-only access: float val = tensor.at({1, 2});
    const float* at(const std::vector<size_t>& indices) const;

    float* begin();
    const float* begin() const ;

    // Returns pointer to the element past the end of the array
    float* end();
    const float* end() const ;
};