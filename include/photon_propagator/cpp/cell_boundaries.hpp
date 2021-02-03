#pragma once

#include <photon_propagator/geometry.hpp>
#include <photon_propagator/configuration.hpp>
#include <photon_propagator/optical_module_lines.hpp>

class CellBoundaries{  
public:

  Cells();
    
  void to_device();
  cells* __device_ptr;
    
private:  
   
  void configure(const Geometry& geometry,
		 const Configuration& configuration,
		 const OpticalModuleLines& optical_module_lines);  
};

