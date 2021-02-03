#pragma once

#include <map>
#include <boost/filesystem.hpp>

#include <photon_propagator/geometry.hpp>
#include <photon_propagator/configuration.hpp>
#include <photon_propagator/device_structs/optical_module_line.h>

class OpticalModuleLines{  
public:

  OpticalModuleLines(const Geometry& geometry, const Configuration& configuration);
    
  void to_device();
  optical_module_line* __device_ptr;
  
  // need to pass this to the kernel somehow
  // currently in configuration.rx
  float largest_line_radius() const;
  
  const std::map<int, optical_module_line>&
  get_optical_module_lines(){ return optical_module_lines_; }

  void pprint() const ;

private:  
  
  // sin12 and cb can be compile time constants
  // there is no need to calculate these on the fly
  // everything is known at compile time.
  // they only change if DIR1 and DIR2 change.
  // for now they're calculated immediately in the ctor,
  // hopefully indicating they have no dependencies.
  float sin12_;  
  float cb_[2][2];
  
  float cl_[2] = {0., 0.};
  float crst_[2] = {0., 0.};
  
  unsigned char is_[CX][CY] = { // 21 x 19
    {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'}, // 1
    {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'}, // 2
    {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0',}, // 3
    {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'}, // 4
    {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'}, // 5
    {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'}, // 6
    {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'}, // 7
    {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'}, // 8
    {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'}, // 9
    {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'}, // 10
    {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'}, // 11
    {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'}, // 12
    {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'}, // 13
    {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'}, // 14 
    {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'}, // 15
    {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'}, // 16
    {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'}, // 17
    {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'}, // 18
    {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'}, // 19
    {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'}, // 20
    {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'} // 21
  };
  
  std::vector<unsigned char> ls_; // Not on the device yet. size is NSTR.
  std::vector<optical_module_line> om_lines_; // host version of what's on the device.
  std::map<int, optical_module_line> optical_module_lines_;
 
  void configure(const Geometry& geometry, const Configuration& configuration);

  void calculate_sin12();
  void calculate_cb();
  
  void configure_cells(const Geometry& geometry,
		       const Configuration& configuration);

};

