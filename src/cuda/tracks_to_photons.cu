#include <curand_kernel.h>

#include "util.cu"
#include "gamma.cu"
#include "rotate.cu"

__global__ void generate_photons(curandState_t* rng_state,				 
				 ices* ice_model,
				 photon* photons,
				 unsigned int photon_buffer_size,
				 track* tracks,
				 unsigned int track_buffer_size){

  const unsigned thread_index = blockIdx.x + threadIdx.x * gridDim.x;
  /* Copy state to local memory for efficiency */
  curandState_t local_rng_state = rng_state[thread_index];
  curandState_t* state_ptr = &local_rng_state;
    
  // this block initializes photons and spreads them
  // in space depending on whether they're from
  // muon tracks or cascades.  the spread is determined
  // from various parameters:
  //   l,f for tracks.
  //   a,b for cascades.

  // OCV == 1 over the speed of light 'c' in a vacuum.
  const float OCV=1./0.299792458; 
  unsigned grid_stride = gridDim.x * blockDim.x;
  for(unsigned int i=thread_index;
      i<photon_buffer_size;
      i += grid_stride){
    
      track trk = tracks[i];      
      float l=trk.l * curand_uniform(state_ptr);
      
      float4 photon_pos=trk.r; // track position
      float3 photon_dir=trk.n; // track direction

      photon_pos.w += OCV*l;
      photon_pos.x += n.x*l;
      photon_pos.y += n.y*l;
      photon_pos.z += n.z*l;

      if(trk.f<curand_uniform(state_ptr)){
	const float a=0.39f;
	const float b=2.61f;
	const float I=1-expf(-b*exp2f(a));
	float cs=max(1-powf(-logf(1-curand_uniform(state_ptr)*I)/b, 1/a), -1.0f);
	float si=sqrtf(1-cs*cs);
	rotate(cs, si, n, state_ptr);
      }

      // different ice layers have different indices of refraction
      // and therefore different cherenkov cones.
      unsigned int j=min(__float2int_rd(N_WAVELENGTH_SLICES*curand_uniform(state_ptr)), N_WAVELENGTH_SLICES-1);
      const ices& w = ice_model[j];      
      rotate(w.coschr, w.sinchr, n, state_ptr);

      // Create a photon and add it to the buffer.
      photon ph;
      ph.r = photon_pos;
      ph.n = photon_dir;
      ph.q = j; // q isn't a track index  it's a wavelength slice index.
      photon_buffer[i] = ph;
  }
}

