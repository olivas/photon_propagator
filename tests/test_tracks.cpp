#include <cassert>

#include <iostream>
#include <memory>

#include <boost/filesystem.hpp>

#include <photon_propagator/cpp/device.hpp>
#include <photon_propagator/cpp/tracks.hpp>

using std::cerr;
using std::endl;

using boost::filesystem::path;
using boost::filesystem::exists;

int main(int argc, char* argv[]){

  std::shared_ptr<Device> device(new Device(0));
  
  particle muon;
  muon.major_id=0;
  muon.minor_id=0;
  muon.ptype=particle_type::muon;
  muon.time=0.;
  muon.energy=100.; // in god-given units
  muon.length=1.;   // in god-given units
  muon.direction={0,0,0};
  muon.position={0,0,0};

  const size_t NTRACKS{1000};
  std::vector<particle> particles;
  particles.assign(NTRACKS, muon);
  
  Tracks tracks(NTRACKS, device);
  for(auto p: particles)
    tracks.add(p);
  tracks.to_device();

  std::cerr<<"tracks.__device_ptr = "<<tracks.__device_ptr<<std::endl;
  std::cerr<<"tracks.n_photons() = "<<tracks.n_photons()<<std::endl;
  
}
  
