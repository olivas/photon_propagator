#include <cassert>

#include <iostream>
#include <boost/filesystem.hpp>

#include <photon_propagator/device.hpp>
#include <photon_propagator/geometry.hpp>

using std::cerr;
using std::endl;

using boost::filesystem::path;
using boost::filesystem::exists;

int main(int argc, char* argv[]){

  if(argc != 2){
    std::cerr<<"Usage: test_geometry <GEOMETRY_FILE>"<<std::endl;
    std::exit(-1);
  }
  
  path geometry_filename(argv[1]);
  assert(exists(path(geometry_filename)));

  Device device(0);
  
  Geometry geometry(geometry_filename);
  geometry.to_device();
  
  const Geometry::packed_map_t& geo = geometry.get_packed_geometry();
  for(auto p: geo){
    unsigned dom{0};
    for(auto pos: p.second){
      std::cerr<<" "<<pos.index<<": ["<<p.first<<","<<dom<<"] = ("
	       <<pos.x<<","
	       <<pos.y<<","
	       <<pos.z<<")\n";
      dom++;
    }
  }

  std::cerr<<"max RDE = "<<geometry.get_max_rde()<<std::endl;
  std::cerr<<"scaled eff = "<<0.75*geometry.get_max_rde()<<std::endl;

}

  
