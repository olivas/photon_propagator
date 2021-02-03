#pragma once

#include <vector>
#include <memory>

#include <photon_propagator/device.hpp>
#include <photon_propagator/particle.hpp>

struct track;

class Tracks{  
public:

  Tracks(const size_t count, const std::shared_ptr<Device>& device);

  ~Tracks();
  
  void add(const particle& p);
  
  size_t nbytes() const;

  size_t n_photons() const;

  void to_device();
  
  track* __device_ptr;
  
  void pprint() const {};

private:  

  size_t count_;
  std::vector<track> host_tracks_;
  const std::shared_ptr<Device>& device_;
   
};

