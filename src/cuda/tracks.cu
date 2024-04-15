#include <iostream>
#include <sstream>

#include <photon_propagator/cpp/tracks.hpp>
#include <photon_propagator/cpp/particle.hpp>
#include <photon_propagator/cpp/photon_yield.hpp>
#include <photon_propagator/cuda/device_structs/track.cuh>
#include <photon_propagator/cuda/check_error.cuh>

using std::stringstream;
using std::string;
using std::cerr;
using std::endl;
using std::shared_ptr;

Tracks::Tracks(const size_t count,
	       const shared_ptr<Device>& device):
  count_(count),
  device_(device)
{
  CHECK_ERROR(cudaMalloc((void**) &__device_ptr, count*sizeof(track)));
};

size_t
Tracks::nbytes() const
{
  return host_tracks_.size()*sizeof(track);
}

size_t Tracks::n_photons() const{
  size_t result{0};
  for(auto t: host_tracks_){
    result += t.q;
  }
  return result;
}

void Tracks::to_device(){
  if(host_tracks_.size() <= count_){
    unsigned long sizeof_track_buffer{host_tracks_.size()*sizeof(track)};
    CHECK_ERROR(cudaMemcpy(__device_ptr,
			   host_tracks_.data(),
			   sizeof_track_buffer,
			   cudaMemcpyHostToDevice));
  }else{
    cerr<<"ERROR: allocated "<<count_<<" pushing "<<host_tracks_.size()<<"."<<endl;	
  }
}

void Tracks::add(const particle& p){
  
  track trk;
  trk.q = photon_yield::bare_muon::yield(p.energy, p.length);
  trk.r.x = p.position[0];
  trk.r.x = p.position[1];
  trk.r.x = p.position[2];
  trk.r.w = p.time;
  trk.n.x = p.direction[0];
  trk.n.y = p.direction[1];
  trk.n.z = p.direction[2];

  trk.l = p.length;
  trk.f = 1./photon_yield::bare_muon::sub_threshold_cascades_fraction(p.energy);
  host_tracks_.push_back(trk);
}


Tracks::~Tracks(){
  CHECK_ERROR(cudaFree(__device_ptr));
};
