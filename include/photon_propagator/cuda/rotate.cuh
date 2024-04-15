#ifndef PHOTON_PROPAGATOR_CUDA_ROTATE_H
#define PHOTON_PROPAGATOR_CUDA_ROTATE_H

#include <curand_kernel.h>

#include <photon_propagator/cuda/swap.cuh>

const float FPI{3.141592653589793};

__device__
void rotate(const float cs, // cosine
            const float si, // sine
            float3& n,      // direction
            curandState_t* state_ptr);

#endif // PHOTON_PROPAGATOR_CUDA_ROTATE_H
