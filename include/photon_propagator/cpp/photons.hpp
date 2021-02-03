#pragma once

#include <vector>
#include <memory>

#include <photon_propagator/device.hpp>

struct photon;

class Photons{  
public:

  Photons(const size_t count, const std::shared_ptr<Device>& device);

  ~Photons();

  struct Position{
    float x,y,z,t;
  };
  
  struct Direction{
    float x,y,z;
  };
  
  void add(const Position&,
	   const Direction &);
  
  size_t nbytes() const;

  void to_device();   // i might prefer push
  
  void from_device(); // i might prefer pull
  
  photon* __device_ptr;
  
  void pprint() const {};

private:  

  size_t count_;
  std::vector<photon> host_photons_;
  const std::shared_ptr<Device>& device_;
   
};

