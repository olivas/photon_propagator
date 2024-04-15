#include <iostream>
#include <sstream>

#include <photon_propagator/cpp/cascades.hpp>
#include <photon_propagator/cpp/particle.hpp>
#include <photon_propagator/cpp/photon_yield.hpp>
#include <photon_propagator/cuda/device_structs/cascade.cuh>
#include <photon_propagator/cuda/check_error.cuh>

using std::stringstream;
using std::string;
using std::cerr;
using std::endl;
using std::shared_ptr;

Cascades::Cascades(const size_t count,
		   const shared_ptr<Device>& device):
  count_(count),
  device_(device)
{
  CHECK_ERROR(cudaMalloc((void**) &__device_ptr, count*sizeof(cascade)));
};

size_t
Cascades::nbytes() const
{
  return host_cascades_.size()*sizeof(cascade);
}

const cascade& Cascades::at(size_t idx) const {
  return host_cascades_.at(idx);
}


size_t Cascades::n_photons() const{
  size_t result{0};
  for(auto c: host_cascades_){
    result += c.q;
  }
  return result;
}

void Cascades::to_device(){
  if(host_cascades_.size() <= count_){
    unsigned long sizeof_cascade_buffer{host_cascades_.size()*sizeof(cascade)};
    CHECK_ERROR(cudaMemcpy(__device_ptr,
			   host_cascades_.data(),
			   sizeof_cascade_buffer,
			   cudaMemcpyHostToDevice));
  }else{
    cerr<<"ERROR: allocated "<<count_<<" pushing "<<host_cascades_.size()<<"."<<endl;	
  }
}

void Cascades::add(const particle& p){
  
  int itype{static_cast<int>(p.ptype)};
  
  cascade cscd;
  cscd.q = photon_yield::cascade::yield(p.energy, itype);
  cscd.r.x = p.position[0];
  cscd.r.x = p.position[1];
  cscd.r.x = p.position[2];
  cscd.r.w = p.time;
  cscd.n.x = p.direction[0];
  cscd.n.y = p.direction[1];
  cscd.n.z = p.direction[2];
  
  int i{static_cast<int>(p.ptype)};
  std::pair<float, float> parameters = photon_yield::cascade::longitudinal_profile_parameters(p.energy, i);
  cscd.a = parameters.first;
  cscd.b = parameters.second;
  
  host_cascades_.push_back(cscd);
}

Cascades::~Cascades(){
  CHECK_ERROR(cudaFree(__device_ptr));
};
