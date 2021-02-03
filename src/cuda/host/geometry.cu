#include <iostream>
#include <sstream>
#include <fstream>

#include <boost/filesystem.hpp>

#include <photon_propagator/geometry.hpp>
#include <photon_propagator/cuda/check_error.cuh>

using std::ifstream;
using std::stringstream;
using std::string;
using std::cerr;
using std::endl;
using std::getline;
using boost::filesystem::path;

namespace{
  unsigned remapped_string_number(const Geometry::OMKey& omkey){
    // the deepcore DOMs are effectively
    // on different strings.
    // they do have different spacings
    // the deep DOMs (n.dom > 10) are
    // shifted to different string numbers.
    // 79 -> 89
    // 80 -> 90
    // 81 -> 91
    // 82 -> 92
    // 83 -> 93
    // 84 -> 94
    // 85 -> 95
    // 86 -> 96
    if(omkey.first > 78 && omkey.second > 10)
      return omkey.first + 10;
    return omkey.first;
  }
}

void
Geometry::configure(const path& geometry_filename){
  string line;
  // the format is str, om, x, y, z, hv, rde
  ifstream infile(geometry_filename.c_str());
  while(getline(infile, line)){
    stringstream sstr(line);
    OpticalModule om;
    OMKey k;
    sstr >> k.first;
    sstr >> k.second;
    sstr >> om.position.x;
    sstr >> om.position.y;
    sstr >> om.position.z;
    sstr >> om.hv;
    sstr >> om.rde;
    detector_[k] = om;
  }    
  cerr<<"Loaded "<<detector_.size()<<" modules into the geometry."<<endl;

  // should be sorted by increasing string number
  // std::pair compare behaves as you would expect,
  // comparing std::pair.first and then second if first's are equal.
  
  for(auto p: detector_){
    auto om = p.second;
    auto omkey = p.first;
    auto string_number = remapped_string_number(omkey);
    Geometry::packed_key_t key(string_number);
    if(packed_geometry_.find(key) == packed_geometry_.end()){
      packed_geometry_[key] = std::vector<Geometry::Position>();
    }
    packed_geometry_[key].push_back(om.position);
  }
  
  // need to set the index into the geometry array.
  // the second member of the key is the index
  // of the string's first DOM in the DOM array.
  size_t index{0};
  for(auto& p : packed_geometry_){

    // sort by decreasing depth
    auto& positions = p.second;
    std::sort(begin(positions), end(positions),
	      [&](Geometry::Position p1, Geometry::Position p2){
		return p1.z > p2.z;
	      });

    for(auto& position: positions){
      position.index = index;
      index++;
    }
  }
}  

float Geometry::get_max_rde(){
  std::vector<float> rdes;
  for(auto p: detector_){
    rdes.push_back(p.second.rde);
  }
  return *std::max_element(begin(rdes), end(rdes));
}


void Geometry::to_device(){
  // FIXME:  The layout in memory has to be solid here.
  for(auto p: detector_){
    DOM d;
    d.r[0] = p.second.position.x;
    d.r[1] = p.second.position.y;
    d.r[2] = p.second.position.z;
    host_doms_.push_back(d);
  }

  unsigned long sizeof_geometry{host_doms_.size()*sizeof(DOM)};
  std::cerr<<"allocated "<<sizeof_geometry
	   <<" bytes ("<<int(sizeof_geometry/1e3)<<" kB) for geometry.\n";
  CHECK_ERROR(cudaMalloc((void**) &__device_ptr, sizeof_geometry));
  CHECK_ERROR(cudaMemcpy(__device_ptr, host_doms_.data(), sizeof_geometry, cudaMemcpyHostToDevice));  
}