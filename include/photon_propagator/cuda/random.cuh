#pragma once

#include <memory>
#include <vector>

#include <curand_kernel.h>

class Device;

class Random {
public:

    Random(const std::shared_ptr <Device> &device,
           const unsigned number_of_blocks,
           const unsigned threads_per_block);

    ~Random() {};

    size_t nbytes() const;

    void initialize(unsigned long long seed,
                    unsigned long long offset);

    void uniform(std::vector<float> &result);

    void random(std::vector<unsigned> &result);

    void gamma(std::vector<float> &result, const float k);

private:

    curandState_t* __device_ptr;
    unsigned number_of_blocks_;
    unsigned threads_per_block_;
    const std::shared_ptr <Device> &device_;
};

