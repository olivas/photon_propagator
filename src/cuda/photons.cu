#include <iostream>
#include <sstream>

#include <photon_propagator/cpp/photons.hpp>
#include <photon_propagator/cuda/device_structs/photon.cuh>
#include <photon_propagator/cuda/check_error.cuh>

using std::stringstream;
using std::string;
using std::cerr;
using std::endl;
using std::shared_ptr;

Photons::Photons(const size_t count,
		 const shared_ptr<Device>& device):
  count_(count),
  device_(device)
{
  CHECK_ERROR(cudaMalloc((void**) &__device_ptr, count*sizeof(photon)));
};

size_t
Photons::nbytes() const
{
  return host_photons_.size()*sizeof(photon);
}

void Photons::to_device(){
  if(host_photons_.size() <= count_){
    unsigned long sizeof_photon_buffer{host_photons_.size()*sizeof(photon)};
    CHECK_ERROR(cudaMemcpy(__device_ptr,
			   host_photons_.data(),
			   sizeof_photon_buffer,
			   cudaMemcpyHostToDevice));
  }else{
    cerr<<"ERROR: allocated "<<count_<<" pushing "<<host_photons_.size()<<"."<<endl;	
  }
}

void Photons::from_device(){
  if(host_photons_.size() <= count_){
    unsigned long sizeof_photon_buffer{host_photons_.size()*sizeof(photon)};
    CHECK_ERROR(cudaMemcpy(host_photons_.data(),
			   __device_ptr,			   
			   sizeof_photon_buffer,
			   cudaMemcpyDeviceToHost));
  }else{
    cerr<<"ERROR: allocated "<<count_<<" pushing "<<host_photons_.size()<<"."<<endl;	
  }
}

void Photons::add(const Position& position,
		  const Direction & direction){

  photon p;
  p.q = 0;
  p.r.x = position.x;
  p.r.x = position.y;
  p.r.x = position.z;
  p.r.w = position.t;
  p.n.x = direction.x;
  p.n.y = direction.y;
  p.n.z = direction.z;
  host_photons_.push_back(p);
}

Photons::~Photons(){
  CHECK_ERROR(cudaFree(__device_ptr));
};
