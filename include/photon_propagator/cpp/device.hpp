#pragma once

class Device{
public:
  Device(int device_number);
  ~Device();

  int multi_processor_count() const { return multi_processor_count_; }
  int total_global_mem() const { return total_global_mem_; }
  
private:
  int device_number_;
  int multi_processor_count_;
  size_t total_global_mem_;
};
