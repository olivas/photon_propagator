#include <cassert>

#include <iostream>
#include <boost/filesystem.hpp>

#include <photon_propagator/cpp/device.hpp>
#include <photon_propagator/cpp/geometry.hpp>
#include <photon_propagator/cpp/configuration.hpp>
#include <photon_propagator/cpp/optical_module_lines.hpp>

using std::cerr;
using std::endl;

using boost::filesystem::path;
using boost::filesystem::exists;

int main(int argc, char* argv[]){

  if(argc != 3){
    std::cerr<<"Usage: test_geometry <GEOMETRY_FILE> <ICE_MODEL_PATH>"<<std::endl;
    std::exit(-1);
  }

  path geometry_filename(argv[1]);
  assert(exists(path(geometry_filename)));
  
  path ice_model_path(argv[2]);
  assert(exists(path(ice_model_path)));

  Device device(0);
  
  Geometry geometry(geometry_filename);
  Configuration config(ice_model_path);

  OpticalModuleLines om_lines(geometry, config);
  om_lines.pprint();

}

  
