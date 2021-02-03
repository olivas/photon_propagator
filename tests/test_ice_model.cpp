#include <cassert>

#include <iostream>
#include <boost/filesystem.hpp>

#include <photon_propagator/device.hpp>
#include <photon_propagator/configuration.hpp>
#include <photon_propagator/ice_model.hpp>

using std::cerr;
using std::endl;

using boost::filesystem::path;
using boost::filesystem::exists;

int main(int argc, char* argv[]){

  if(argc != 2){
    std::cerr<<"Usage: test_ice_model <ICE_MODEL_PATH>"<<std::endl;
    std::exit(-1);
  }

  path ice_model_path(argv[1]);
  assert(exists(path(ice_model_path)));
  
  Configuration config(ice_model_path);
  float scattering_angle{config.get_configuration().g};
  
  IceModel ice_model(ice_model_path, scattering_angle);

  Device device(0);
  ice_model.to_device();

  ice_model.pprint();
}

  
