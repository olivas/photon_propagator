# pragma once

#include <boost/filesystem.hpp>
#include <photon_propagator/device_structs/configuration.h>

class Configuration{
public:
  // We're precalculating a lot of things that are
  // accessed from global (or potentially shared)
  // memory.  This is expensive on the GPU.
  // Is this to relieve register pressure
  // for more concurrent kernel launches?
  // ...maybe.
  // Doubt it, but maybe.  Should investigate.
  // Is the trade off worth it?  This is the real question.
  // How many registers are available?
  // Should these be precalculated in the beginning?
  // Or maybe stored in constant memory.
  //d.ocv = 1/C;
  
  Configuration(const boost::filesystem::path& ice_model_path);

  // only expose needs to be exposed
  const configuration& get_configuration() const { return host_configuration_; }
  float get_xR() const { return xR_; }
  void pprint();
  
  //d.size = dp.size();
  //d.dh = dp[1]-dp[0];
  //d.hdh = dh/2;
  //d.rdh = 1/dh;
  
  void to_device();
  configuration* __device_ptr;
private:
  
  boost::filesystem::path configuration_filename_;
  boost::filesystem::path angular_sensitivity_filename_;

  float xR_;
  configuration host_configuration_;
  
  void configure(const boost::filesystem::path& configuration_filename);
  void configure_angular_sensitivity(const boost::filesystem::path& angular_sensitivity_filename);
};
