#include <curand_kernel.h>


// this is getting confusing
// this is the kernel that we're going to launch.
#include <photon_propagator/cpp/device.hpp>
#include <photon_propagator/cpp/ice_model.hpp>
#include <photon_propagator/cpp/cascades.hpp>
#include <photon_propagator/cpp/photons.hpp>
#include <photon_propagator/cpp/cascades_to_photons.hpp>
#include <photon_propagator/cuda/random.cuh>
#include <photon_propagator/cuda/cascades_to_photons.cuh>

void cascades::GeneratePhotons(const Random& rng,
			       const IceModel& ice_model,
			       const Cascades& cascades,
			       const std::shared_ptr<Device>& device,
			       const unsigned number_of_blocks,
			       const unsigned threads_per_block,
			       Photons& output){
  
  // setup
  // we can only launch in blocks of n_concurrent_threads.
  // we have lots of photons.
  // this is why it makes sense to generate, propagate, and pull hits.    

  const unsigned n_concurrent_threads{number_of_blocks*threads_per_block};
  const unsigned n_photons{n_concurrent_threads};
  Photons photons(n_concurrent_threads, device);
  photons.to_device(); // i don't need to add photons.  the kernel does that.
                       // in fact this is the whole point of this kernel
                       // to set the initial (pre-propagation) state of the
                       // photons.
  
  const cascade& cscd = cascades.at(0);
  cascades_to_photons<<<number_of_blocks, threads_per_block>>>(rng.__device_ptr,
							       ice_model.__device_ptr,
							       photons.__device_ptr,
							       n_photons,
							       cscd);
  
  // Now pull the photons off the device
  // and fill the 'output' vector.
  
  // it might actually make more sense at some point to do as clsim does
  // and have each kernel propagate N photons.
}

