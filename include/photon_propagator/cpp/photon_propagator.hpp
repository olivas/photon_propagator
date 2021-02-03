#pragma once

#include <vector>
#include <array>
#include <map>

#include <photon_propagator/detector.hpp>
#include <photon_propagator/particle.hpp>

struct OM;
using event = std::vector<particle>;
using domkey = std::pair<int, int>;
using photon_arrival_times = std::map<domkey, std::vector<double>>;

class photon_propagator{

public:
  photon_propagator(const detector&);
  ~photon_propagator();

  void process_event(const event&, photon_arrival_times&);
  
private:
      
  void propagate_photons(const particle&, unsigned);

  bool isinside(const particle&) const;

  unsigned event_number_; 
  int gpu_number_;
  std::vector<OM> i3oms_;

};
