#pragma once

#include <vector>
#include <memory>

#include <photon_propagator/device.hpp>
#include <photon_propagator/particle.hpp>

struct cascade;

class Cascades{  
public:

  Cascades(const size_t count, const std::shared_ptr<Device>& device);

  ~Cascades();
  
  void add(const particle& p);
  
  size_t nbytes() const;

  size_t n_photons() const;
  
  void to_device();
  
  cascade* __device_ptr;
  
  void pprint() const {};

  const cascade& at(size_t idx) const;
  
private:  

  size_t count_;
  std::vector<cascade> host_cascades_;
  const std::shared_ptr<Device>& device_;
   
};

