#pragma once

#include <vector>
 
// for configuration what's needed is
// position(x,y,z), HV, and RDE

struct optical_module{
  int str;
  unsigned dom;
  double x, y, z;
  double hv;
  double rde;
};

using detector = std::vector<optical_module>;
  
