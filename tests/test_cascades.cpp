#include <cassert>

#include <iostream>
#include <memory>

#include <boost/filesystem.hpp>

#include <photon_propagator/cpp/device.hpp>
#include <photon_propagator/cpp/particle.hpp>
#include <photon_propagator/cpp/cascades.hpp>

using std::cerr;
using std::endl;

using boost::filesystem::path;
using boost::filesystem::exists;

int main(int argc, char* argv[]){

  std::shared_ptr<Device> device(new Device(0));

  const size_t NPARTICLES{100};

  particle electron;
  electron.major_id=0;
  electron.minor_id=0;
  electron.ptype=particle_type::e_minus;
  electron.time=0.;
  electron.energy=100.; // in god-given units
  electron.length=0.;  
  electron.direction={0,0,0};
  electron.position={0,0,0};

  std::vector<particle> particles;
  particles.assign(NPARTICLES, electron);
  
  Cascades cascades(NPARTICLES, device);
  for(auto p: particles)
    cascades.add(p);
  cascades.to_device();

  std::cerr<<"cascades.__device_ptr = "<<cascades.__device_ptr<<std::endl;
  std::cerr<<"cascades.n_photons() = "<<cascades.n_photons()<<std::endl;
  
}
  
