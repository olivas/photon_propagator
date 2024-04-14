#pragma once

#include <map>
#include <boost/filesystem.hpp>

#include <photon_propagator/cuda/device_structs/dom.cuh>

class Geometry{  
public:

  Geometry(const boost::filesystem::path& geometry_filename):
    geometry_filename_(geometry_filename)
  {
    configure(geometry_filename);
  };
  
  struct Position{
    unsigned index;
    float x;
    float y;
    float z;
  };
  
  struct OpticalModule{
    Position position;
    float hv;
    float rde;
  };
  
  using OMKey = std::pair<int, int>;

  // the first value in the key is the string number
  // the second value in the key is the index into the DOM array.
  using packed_key_t = unsigned;
  using packed_value_t = std::vector<Geometry::Position>;  
  using packed_map_t = std::map<packed_key_t, packed_value_t>;
  
  // the DOM array is ordered by increasing string number
  void to_device();
  DOM* __device_ptr;

  size_t size_of() const { return host_doms_.size()*sizeof(DOM); }
  
  // this is a convenience method provided for calculating
  // cylinder properties for the optical module lines.
  float get_max_rde();
  const std::map<packed_key_t, packed_value_t>&
  get_packed_geometry() const { return packed_geometry_; }
  
private:

  // load this and push to the device
  std::vector<DOM> host_doms_;
  
  // ordered by increasing string number
  std::map<packed_key_t, packed_value_t> packed_geometry_;
  
  boost::filesystem::path geometry_filename_;
  std::map<OMKey, OpticalModule> detector_;
 
  void configure(const boost::filesystem::path& geometry_filename);  
};

