#pragma once

#include <map>
#include <boost/filesystem.hpp>

#include <photon_propagator/geometry.hpp>
#include <photon_propagator/device_structs/optical_module_line.h>

class Cells{  
public:

  Cells();
    
  void to_device();
  cells* __device_ptr;
    
private:  
  
  // load this and push to the device
  std::vector<optical_module_line> om_lines_;

  std::map<int, optical_module_line> optical_module_lines_;
 
  void configure(const Geometry& geometry, float dom_radius);  
};

