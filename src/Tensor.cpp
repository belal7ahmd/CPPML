
#include "Tensor.hpp"

Tensor::Tensor() : data(nullptr) {};

Tensor::Tensor(std::vector<size_t> tensor_shape) {
    shape = tensor_shape;

    size = 1;

    is_owner = true;

    for (int axis: tensor_shape) {
        size *= axis;
    }

    data = new float[size];

    strides.resize(shape.size());

    size_t current_stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = current_stride;
        current_stride *= shape[i];
    }

};

Tensor::Tensor(std::initializer_list<float> values, std::vector<size_t> tensor_shape)
    : Tensor(tensor_shape) {
        
    // Safety check: copy only up to the allocated size to prevent crashes
    size_t elements_to_copy = std::min(size, values.size());
    std::copy(values.begin(), values.begin() + elements_to_copy, data);
}

Tensor::~Tensor() {
    // Double freeing Safety gaurd
    if (is_owner) {
        delete[] data;
    }
};

Tensor& Tensor::operator=(const Tensor& source) {
    if (this == &source) {
        return *this;
    }

    shape = source.shape;

    size = source.size;

    strides = source.strides;

    is_owner = (size > 0);

    if (size <= 0) {
        data = nullptr;
        return *this;
    }

    delete[] data;

    data = new float[size];

    const float* source_data = source.begin();

    for (size_t i = 0; i < size; i++, source_data++) {
        data[i] = *source_data;
    };

    return *this;
};

Tensor::Tensor(const Tensor& source) {

    shape = source.shape;

    size = source.size;

    strides = source.strides;

    is_owner = (size > 0);

    if (size <= 0) {
        data = nullptr;
        return;
    }

    data = new float[size];

    const float* source_data = source.begin();

    for (size_t i = 0; i < size; i++, source_data++) {
        data[i] = *source_data;
    };
};


void Tensor::fill(float new_val) {
    for (float& val: *this) {
        val = new_val;
    }
}


void Tensor::matmul(const Tensor& mat_a, const Tensor& mat_b, Tensor& dst) {
    
    if (mat_a.shape.size() != 2 || mat_b.shape.size() != 2) {
        throw std::invalid_argument("Tensors for mat mul should have two axis use tensormul instead");
        return;
    }

    if (mat_a.shape[1] != mat_b.shape[0]) {
        throw std::invalid_argument("Inner dimensions for matmul must match");
        return;
    }

    dst.fill(0.0f);

    // Standard i-k-j Loop
    for (size_t i = 0; i < mat_a.shape[0]; i++) {
        for (size_t k = 0; k < mat_a.shape[1]; k++) {
            
            for (size_t j = 0; j < mat_b.shape[1]; j++) {
                *dst.at({i, j}) += (*mat_a.at({i, k})) * (*mat_b.at({k, j}));
            }
        }
    }


};

void Tensor::sum(const Tensor& tensor_a, const Tensor& tensor_b, Tensor& dst) {
    if (tensor_a.shape != tensor_b.shape) {
        throw std::invalid_argument("Tensors much match shape for summing");
        return;
    }

    for (int i = 0; i < tensor_a.size; i++) {
        dst.data[i] = tensor_a.data[i] + tensor_b.data[i];
    }

};

void Tensor::multiply(const float scalar) {
    for (float& val: *this) {
        val *= scalar;
    }
};

void Tensor::subtract(const Tensor& tensor_b) {
    if (shape != tensor_b.shape) {
        throw std::invalid_argument("Tensors much match shape for subtracting");
        return;
    }

    for (int i = 0; i < size; i++) {
        data[i] -= tensor_b.data[i];
    }

};

void Tensor::transpose() {
    if (shape.size() != 2) return;


    float* new_data = new float[size];
    std::vector<size_t> new_shape = {shape[1], shape[0]};

    for (size_t r = 0; r < shape[0]; ++r) {
        for (size_t c = 0; c < shape[1]; ++c) {
            size_t old_idx = strides[0] * r + strides[1] * c;
            size_t new_idx = c * shape[0] + r;
            new_data[new_idx] = data[old_idx];
        }
    }

    delete[] data;
    data = new_data;
    shape = new_shape;
    
    strides[0] = shape[0];
    strides[1] = 1;
}

void Tensor::stack(const std::vector<Tensor>& tensors, Tensor& dst) {
    if (tensors.empty()) return;

    // 1. Validate that all input tensors have the exact same size
    size_t single_tensor_size = tensors[0].size; 
    for (const Tensor& t : tensors) {
        if (t.size != single_tensor_size) {
            throw std::runtime_error("All tensors must be the same size to stack!");
        }
    }

    // 2. Ensure the destination tensor is big enough to hold everything
    size_t total_elements = single_tensor_size * tensors.size();

    // 3. Loop and copy the raw data into dst
    float* dst_ptr = dst.begin(); 
    size_t offset = 0;

    for (const Tensor& tensor : tensors) {
        // Copy data from the current tensor into the current position of dst
        std::memcpy(dst_ptr + offset, tensor.begin(), single_tensor_size * sizeof(float));
        
        offset += single_tensor_size;
    }
}



void Tensor::rand_init(float min_range, float max_range) {
    // Static ensures the generator is only initialized once, not every time the function runs
    static std::random_device rd;
    static std::mt19937 gen(rd()); 
    std::uniform_real_distribution<float> dist(min_range, max_range);

    for (int i = 0; i < size; i++) {
        data[i] = dist(gen);
    }
}


void Tensor::print_tensor() const {
    if (!data) {
        std::cout << "[]\n";
        return;
    }
    
    std::cout << "Tensor Shape: [";
    for(size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << (i < shape.size() - 1 ? ", " : "");
    }
    std::cout << "], Strides: [";
    for(size_t i = 0; i < strides.size(); ++i) {
        std::cout << strides[i] << (i < strides.size() - 1 ? ", " : "");
    }
    std::cout << "]\nData: [";
    for (size_t i = 0; i < size; ++i) {
        if (i > 5 && i < size - 5) {
            continue;
        }
        std::cout << data[i] << (i < size - 1 ? ", " : "");
    }
    std::cout << "]\n\n";
}

// For modifying data: tensor.at({1, 2}) = 5.0f;
float* Tensor::at(const std::vector<size_t>& indices) {
    size_t index = 0;
    for (int i = 0; i < strides.size(); i++) {
        index += strides[i] * indices[i];
    }

    return data + index;
};

// For read-only access: float val = tensor.at({1, 2});
const float* Tensor::at(const std::vector<size_t>& indices) const {
    size_t index = 0;
    for (int i = 0; i < strides.size(); i++) {
        index += strides[i] * indices[i];
    }

    return data + index;
};

float* Tensor::begin() { return data; }
const float* Tensor::begin() const { return data; }

// Returns pointer to the element past the end of the array
float* Tensor::end() { 
    return data + size; 
}

const float* Tensor::end() const {
    return data + size; 
}
