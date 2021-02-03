#include <curand_kernel.h>

#include <photon_propagator/cuda/check_error.cuh>
#include <photon_propagator/cuda/rotate.cuh>
#include <photon_propagator/test_rotate.hpp>


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

void test::test_rotations(std::vector<std::array<float, 3>>& directions,		    
			  const unsigned long long number_of_blocks,
			  const unsigned long long threads_per_block,
			  const std::shared_ptr<Random>& rng,
			  const float cs,
			  const float si){
  unsigned long long n_concurrent_threads{number_of_blocks*threads_per_block}; 
  size_t sizeof_result = n_concurrent_threads * sizeof(float3);
  
  float3* __result;
  CHECK_ERROR(cudaMalloc((void**) &__result, sizeof_result));
  std::cerr<<"allocated "<<sizeof_result<<" bytes ("<<int(sizeof_result/1e3)<<" kB) for result.\n";

  // copy directions into __result
  CHECK_ERROR(cudaMemcpy(__result,
			 directions.data(),    			 
    			 sizeof_result,
    			 cudaMemcpyHostToDevice));
  
  test_kernel_rotate<<<number_of_blocks, threads_per_block>>>(cs, si, rng->__device_ptr, __result);

  // pull the result off the device and fill the result vector.
  CHECK_ERROR(cudaMemcpy(directions.data(),
    			 __result,
    			 sizeof_result,
    			 cudaMemcpyDeviceToHost));
}
