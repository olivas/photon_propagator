#include <iostream>
#include <sstream>

#include <photon_propagator/hits.hpp>
#include <photon_propagator/device_structs/hit.h>
#include <photon_propagator/cuda/check_error.cuh>

Hits::Hits(const size_t hit_buffer_size){
  unsigned long sizeof_hit_buffer{hit_buffer_size*sizeof(hit)};
  std::cerr<<"allocated "<<sizeof_hit_buffer
	   <<" bytes ("<<int(sizeof_hit_buffer/1e3)<<" kB) for hits.\n";
  CHECK_ERROR(cudaMalloc((void**) &__device_ptr, sizeof_hit_buffer));
}

Hits::~Hits(){
  CHECK_ERROR(cudaFree(__device_ptr));
};