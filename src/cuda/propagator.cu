
#include <photon_propagator/cpp/propagator.hpp>
#include <photon_propagator/cpp/photons.hpp>
#include <photon_propagator/cpp/photon_yield.hpp>

#include <photon_propagator/cuda/check_error.cuh>

// Not ready yet.
//#include "propagation_kernel.cu"

Propagator::Propagator(const std::shared_ptr<Device>& device,
		       const Configuration& configuration,
		       const IceModel& ice_model,
		       const Geometry& geometry,
		       const OpticalModuleLines& om_lines,
		       const size_t hit_buffer_size, 
		       const size_t photon_buffer_size):
  device_(device),
  configuration_(configuration),
  ice_model_(ice_model),
  geometry_(geometry),
  om_lines_(om_lines),
  number_of_blocks_(1),
  threads_per_block_(1024),
  host_hits_(hit_buffer_size),
  //host_photons_(photon_size, device),
  time_spent_on_device_(0)
{
          
  CHECK_ERROR(cudaStreamCreate(&stream_));
  CHECK_ERROR(cudaEventCreateWithFlags(&event1_, cudaEventBlockingSync));
  CHECK_ERROR(cudaEventCreateWithFlags(&event2_, cudaEventBlockingSync));
  
  number_of_blocks_ = device_->multi_processor_count();
  cerr<<"Execution configuration <<<"<<number_of_blocks_<<", "<<threads_per_block_<<">>>"<<endl;      

  // Determine how much space we have for hits and photons.
  unsigned n_concurrent_threads{number_of_blocks_*threads_per_block_}; // aka "grid size"  
  unsigned long sizeof_curand{n_concurrent_threads * sizeof(curandState_t)};
  unsigned long base_memory_requirement{ice_model.size_of()+
      sizeof(configuration) +
      geometry.size_of() +
      sizeof_curand};
  
  unsigned available_memory = device_->total_global_mem() - base_memory_requirement;
  std::cerr<<"Base Memory Required = "<<base_memory_requirement<<std::endl;
  std::cerr<<"Available Memory = "<<available_memory<<std::endl;
}

void Propagator::execute(){
  
  CHECK_ERROR(cudaStreamSynchronize(stream_));
  CHECK_ERROR(cudaEventRecord(event1_, stream_));      
  CHECK_ERROR(cudaGetLastError());
  
//  propagate<<< number_of_blocks_, threads_per_block_, 0, stream_ >>>(__device_configuration_ptr,
//								     __device_rng_state,
//								     __device_ice_model_ptr,
//								     __device_geometry_ptr,
//								     __device_photon_buffer_ptr,
//                                                                   __device_photons_ptr,
//								     photon_buffer_size,
//								     __device_hits_ptr,
//								     hit_buffer_size)
//								     
//
  CHECK_ERROR(cudaGetLastError());
  CHECK_ERROR(cudaEventRecord(event2_, stream_));  
  CHECK_ERROR(cudaStreamSynchronize(stream_));
  
//  // pull 'e' off of the device and into 'd'
//  // but only read the first 7 ints worth of data...wtf?
//  // CHECK_ERROR(cudaMemcpy(&host_dats_, __device_dats_ptr, 7*sizeof(int), cudaMemcpyDeviceToHost));
//  // pretty sure we only need the hidx, which is the first int.
//  // maybe the hit_buffer_size, which is the second.
//  // Nope...one is sufficient.  We just need hidx.
  float dt{0.};
//  CHECK_ERROR(cudaMemcpy(&host_dats_, __device_dats_ptr, sizeof(int), cudaMemcpyDeviceToHost));
  CHECK_ERROR(cudaEventElapsedTime(&dt, event1_, event2_));
  time_spent_on_device_ += dt;
  
//  unsigned int size=host_dats_.hidx*sizeof(hit);
//  CHECK_ERROR(cudaMemcpyAsync(&host_sensors_.hits[global_context.hidx],
//			      __device_hits_ptr,
//			      size,
//			      cudaMemcpyDeviceToHost,
//			      stream_));
}

Propagator::~Propagator(){    
  CHECK_ERROR(cudaEventDestroy(event1_));
  CHECK_ERROR(cudaEventDestroy(event2_));
  CHECK_ERROR(cudaStreamDestroy(stream_));  
  std::cerr<<"Device time: "<< time_spent_on_device_ <<" ms\n";	       
}

void
Propagator::fill(const particle& particle, Tracks& tracks, Cascades& cascades){

  int particle_type = static_cast<int>(particle.ptype);
  float eff{1.};
  if(particle.is_muon()){
    // specific to muons
    //Tracks::Position pos{particle.position[0], particle.position[1], particle.position[2], particle.time};
    //Tracks::Direction dir{particle.direction[0], particle.direction[1], particle.direction[2]};
    //Tracks::Parameters params;
    //params.l = particle.length;
    //params.f = 1./photon_yield::bare_muon::sub_threshold_cascades_fraction(particle.energy);
    float n_photons{eff * photon_yield::bare_muon::yield(particle.energy, particle.length)};

    const unsigned photon_bunch_size{10}; // this scales with oversize factor
    unsigned long long photon_yield = llroundf(n_photons);  
    for(unsigned long long photon_count{0};
	photon_count < photon_yield;
	photon_count += photon_bunch_size){
        //tracks.add(pos, dir, params);
        tracks.add(particle);
    }
    
  }else{
    // specific to cascades
    //Cascades::Position pos{particle.position[0], particle.position[1], particle.position[2], particle.time};
    //Cascades::Direction dir{particle.direction[0], particle.direction[1], particle.direction[2]};
    //Cascades::Parameters params;
    
    //std::pair<float, float> long_params = photon_yield::cascade::longitudinal_profile_parameters(particle.energy,
	//											 particle_type);
    //params.a = long_params.first;
    //params.b = long_params.second;

    float n_photons{eff * photon_yield::cascade::yield(particle.energy, particle_type)};

    const unsigned photon_bunch_size{10}; // this scales with oversize factor
    unsigned long long photon_yield = llroundf(n_photons);  
    for(unsigned long long photon_count{0};
	photon_count < photon_yield;
	photon_count += photon_bunch_size){
        //cascades.add(pos, dir, params);
        cascades.add(particle);
    }
    
  }
  
  tracks.to_device();
  cascades.to_device();
}

