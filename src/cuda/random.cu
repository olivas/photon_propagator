
#include <curand_kernel.h>

#include <photon_propagator/cpp/device.hpp>
#include <photon_propagator/cuda/random.cuh>
#include <photon_propagator/cuda/check_error.cuh>
#include <photon_propagator/cuda/forward_declarations.cuh>

__global__ void setup_rng_state(curandState_t *state,
                                unsigned long long seed,
                                unsigned long long offset) {
    // https://docs.nvidia.com/cuda/curand/device-api-overview.html
    //
    // For the highest quality parallel pseudorandom number generation,
    // each experiment should be assigned a unique seed. Within an
    // experiment, each thread of computation should be assigned a
    // unique sequence number. If an experiment spans multiple kernel
    // launches, it is recommended that threads between kernel launches
    // be given the same seed, and sequence numbers be assigned in a
    // monotonically increasing way. If the same configuration of
    // threads is launched, random state can be preserved in global
    // memory between launches to avoid state setup time.

    int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int sequence = thread_idx;
    curand_init(seed, sequence, offset, &state[thread_idx]);
}

__global__ void generate_uniform(curandState_t *state, float *result) {
    int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    /* Copy state to local memory for efficiency */
    curandState localState = state[thread_idx];
    /* Generate pseudo-random unsigned ints */
    result[thread_idx] = curand_uniform(&localState);
    /* Copy state back to global memory */
    state[thread_idx] = localState;
}

__global__ void generate_random(curandState_t *state, unsigned *result) {

    int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    /* Copy state to local memory for efficiency */
    curandState localState = state[thread_idx];
    /* Generate pseudo-random unsigned ints */
    result[thread_idx] = curand(&localState);
    /* Copy state back to global memory */
    state[thread_idx] = localState;
}


__global__ void generate_gamma(float k,
                               curandState_t *state_ptr,
                               float *result) {
    const unsigned thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    /* Copy state to local memory for efficiency */
    curandState localState = state_ptr[thread_idx];
    /* Generate pseudo-random unsigned ints */
    result[thread_idx] = gamma(k, &localState);
    /* Copy state back to global memory */
    state_ptr[thread_idx] = localState;
}

Random::Random(const std::shared_ptr <Device> &device,
               const unsigned number_of_blocks,
               const unsigned threads_per_block) :
        device_(device),
        number_of_blocks_(number_of_blocks),
        threads_per_block_(threads_per_block) {
    unsigned n_concurrent_threads{number_of_blocks_ * threads_per_block_};
    unsigned long sizeof_curand = n_concurrent_threads * sizeof(curandState_t);
    std::cerr << "allocating " << sizeof_curand << " bytes (" << int(sizeof_curand / 1e3) << " kB) for curand state.\n";
    CHECK_ERROR(cudaMalloc((void **) &(__device_ptr), sizeof_curand));
}

size_t Random::nbytes() const {
    unsigned n_concurrent_threads{number_of_blocks_ * threads_per_block_};
    unsigned long sizeof_curand = n_concurrent_threads * sizeof(curandState_t);
    return sizeof_curand;
}

void Random::initialize(unsigned long long seed,
                        unsigned long long offset) {
    std::cerr << "Initializing " << (number_of_blocks_ * threads_per_block_) << " rng states";
    setup_rng_state<<<number_of_blocks_, threads_per_block_>>>(__device_ptr, seed, offset);
    CHECK_ERROR(cudaDeviceSynchronize());
    std::cerr << "...done." << std::endl;
}

void Random::uniform(std::vector<float> &result) {

    unsigned n_concurrent_threads{number_of_blocks_ * threads_per_block_};
    size_t sizeof_result = n_concurrent_threads * sizeof(float);

    float *__result;
    CHECK_ERROR(cudaMalloc((void **) &__result, sizeof_result));
    std::cerr << "allocated " << sizeof_result << " bytes (" << int(sizeof_result / 1e3) << " kB) for result.\n";

    generate_uniform<<<number_of_blocks_, threads_per_block_>>>(__device_ptr, __result);

    result.clear();
    result.reserve(n_concurrent_threads);
    result.assign(n_concurrent_threads, 0);

    // pull the result off the device and fill the result vector.
    CHECK_ERROR(cudaMemcpy(result.data(),
                           __result,
                           sizeof_result,
                           cudaMemcpyDeviceToHost));
}

void Random::random(std::vector<unsigned> &result) {

    unsigned n_concurrent_threads{number_of_blocks_ * threads_per_block_};
    size_t sizeof_result = n_concurrent_threads * sizeof(unsigned);

    unsigned *__result;
    CHECK_ERROR(cudaMalloc((void **) &__result, sizeof_result));
    std::cerr << "allocated " << sizeof_result << " bytes (" << int(sizeof_result / 1e3) << " kB) for result.\n";

    generate_random<<<number_of_blocks_, threads_per_block_>>>(__device_ptr, __result);

    result.clear();
    result.reserve(n_concurrent_threads);
    result.assign(n_concurrent_threads, 0);

    // pull the result off the device and fill the result vector.
    CHECK_ERROR(cudaMemcpy(result.data(),
                           __result,
                           sizeof_result,
                           cudaMemcpyDeviceToHost));
}


void Random::gamma(std::vector<float> &result,
                   const float k) {

    unsigned n_concurrent_threads{number_of_blocks_ * threads_per_block_};
    size_t sizeof_result = n_concurrent_threads * sizeof(float);

    float *__result;
    CHECK_ERROR(cudaMalloc((void **) &__result, sizeof_result));
    std::cerr << "allocated " << sizeof_result << " bytes (" << int(sizeof_result / 1e3) << " kB) for result.\n";

    generate_gamma<<<number_of_blocks_, threads_per_block_>>>(k, __device_ptr, __result);

    result.clear();
    result.reserve(n_concurrent_threads);
    result.assign(n_concurrent_threads, 0);

    // pull the result off the device and fill the result vector.
    CHECK_ERROR(cudaMemcpy(result.data(),
                           __result,
                           sizeof_result,
                           cudaMemcpyDeviceToHost));
}
