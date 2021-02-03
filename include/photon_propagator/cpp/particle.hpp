#pragma once

#include <array>

enum class particle_type{
    unset = 0,
    muon = -1,
    em_shower = 1,
    e_minus = 2,
    e_plus = 3,
    gamma = 4,
    nucl_shower = 101,
    pi_plus = 102,
    pi_minus = 103,
    k_long = 104,
    p_plus = 105,
    neutron = 106,
    p_minus = 107
};

struct particle{
  uint64_t major_id;
  int32_t minor_id;
  particle_type ptype;
  float time;
  float energy;
  float length;
  std::array<float, 3> direction;
  std::array<float, 3> position;

  bool is_muon() const {
    return ptype == particle_type::muon;
  }

  bool is_cascade() const {
    return ~is_muon();
  }

  std::string type_string() const;
};
