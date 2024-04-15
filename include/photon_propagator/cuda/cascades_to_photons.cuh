#pragma once

#include <curand_kernel.h>

#include <photon_propagator/cpp/constants.hpp>
#include <photon_propagator/cuda/device_structs/ice_color.cuh>
#include <photon_propagator/cuda/device_structs/photon.cuh>
#include <photon_propagator/cuda/device_structs/cascade.cuh>
#include <photon_propagator/cuda/gamma.cuh>
#include <photon_propagator/cuda/rotate.cuh>

/**
 * This randomize several things:
 *   1) Photon wavelength
 *   2) Angle for the typical cascades based on GEANT fits (right?)
 *   3) Angle for the cherenkov cone, based on color
 *   4) Position with respect to the cascade
 */

__global__
void cascades_to_photons(curandState_t* rng_state,				 
			 ice_color* ice_model,
			 photon* photons,
			 unsigned int photon_buffer_size,
			 cascade cscd){
  
  const unsigned thread_idx = blockIdx.x + threadIdx.x * gridDim.x;
  /* Copy state to local memory for efficiency */
  curandState_t local_rng_state = rng_state[thread_idx];
    
  // OCV is 1. over the speed of light 'c' in a vacuum.
  const float OCV=1./0.299792458;

  float4 photon_pos = cscd.r; // cascade position
  float3 photon_dir = cscd.n; // cascade direction
  
  unsigned grid_stride = gridDim.x * blockDim.x;
  for(unsigned int i=thread_idx;
      i<photon_buffer_size;
      i += grid_stride){

    // sample the length along the elongated cascade
    float l = cscd.b * gamma(cscd.a, &local_rng_state);
    
    // set the photon position
    photon_pos.w += OCV*l;
    photon_pos.x += cscd.n.x*l;
    photon_pos.y += cscd.n.y*l;
    photon_pos.z += cscd.n.z*l;
    
    // TODO: Where do these magic numbers come from?
    //       They have to be angular parameters from GEANT fits.
    const float a=0.39f;
    const float b=2.61f;
    const float I=1-expf(-b*exp2f(a));
    float cs=max(1-powf(-logf(1-curand_uniform(&local_rng_state)*I)/b, 1/a), -1.0f);
    float si=sqrtf(1-cs*cs);
    rotate(cs, si, cscd.n, &local_rng_state);
    
    // different ice layers have different indices of refraction
    // and therefore different cherenkov cones.
    unsigned int color_idx=min(__float2int_rd(N_WAVELENGTH_SLICES*curand_uniform(&local_rng_state)), N_WAVELENGTH_SLICES-1);
    const ice_color& w = ice_model[color_idx];      
    rotate(w.coschr, w.sinchr, cscd.n, &local_rng_state);
    
    // Create a photon and add it to the buffer.
    photon ph;
    ph.r = photon_pos;
    ph.n = photon_dir;
    ph.q = color_idx; // it's a wavelength slice index.
    photons[i] = ph; // this only works if the number of concurrent threads is
                     // less than the number of photons.
  }
  
  // copy the state back to global memory
  rng_state[thread_idx] = local_rng_state;
}
