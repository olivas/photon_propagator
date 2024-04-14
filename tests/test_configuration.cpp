#include <cassert>

#include <iostream>
#include <boost/filesystem.hpp>

#include <photon_propagator/cpp/device.hpp>
#include <photon_propagator/cpp/configuration.hpp>

using std::cerr;
using std::endl;

using boost::filesystem::path;
using boost::filesystem::exists;

int main(int argc, char* argv[]){

  if(argc != 2){
    std::cerr<<"Usage: test_configuration <ICE_MODEL_PATH>"<<std::endl;
    std::exit(-1);
  }
  
  path ice_model_path(argv[1]);
  assert(exists(path(ice_model_path)));

  Device device(0);

  Configuration config(ice_model_path);
  config.to_device();
  config.pprint();
}

  
