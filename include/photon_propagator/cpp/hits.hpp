#pragma once

#include <vector>

struct hit;

class Hits{  
public:

  // delete the default ctor - rule of 6!!!
  Hits(const size_t hit_buffer_size);
  ~Hits();
  
  hit* __device_ptr;  
  void pprint() const {};

private:  
       
};
