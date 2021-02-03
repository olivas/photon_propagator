#include <cassert>

#include <iostream>
#include <memory>

#include <boost/filesystem.hpp>

#include <photon_propagator/device.hpp>
#include <photon_propagator/photons.hpp>

using std::cerr;
using std::endl;

using boost::filesystem::path;
using boost::filesystem::exists;

int main(int argc, char* argv[]){

  std::shared_ptr<Device> device(new Device(0));

  const size_t NPHOTONS{1000};
  Photons photons(NPHOTONS, device);
  
  Photons::Position pos{0., 1., 2., 3.};
  Photons::Direction dir{4., 5., 6.};
  
  for(unsigned i{0}; i < NPHOTONS; ++i)
    photons.add(pos, dir);
  photons.to_device();

  std::cerr<<"photons.__device_ptr = "<<photons.__device_ptr<<std::endl;
  
}

  
