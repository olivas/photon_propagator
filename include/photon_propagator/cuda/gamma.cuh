#pragma once

#include <curand_kernel.h>

// forward declaration of the gamma device function.
__device__ float gamma(float k, curandState_t* state_ptr);


