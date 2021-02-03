#include <curand_kernel.h>

#include <photon_propagator/cuda/rotate.cuh>

__global__
void test_kernel_rotate(const float cs,       // cosine
			const float si,       // sine
			curandState_t* state, // rng
			float3* result){      // direction
			
  int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
  /* Copy state to local memory for efficiency */
  curandState local_state = state[thread_idx];
  /* Scramble the input vector */
  rotate(cs, si, result[thread_idx], &local_state);
  /* Copy state back to global memory */
  state[thread_idx] = local_state;
}
