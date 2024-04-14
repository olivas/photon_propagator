#include <photon_propagator/cpp/device.hpp>
#include <photon_propagator/cuda/check_error.cuh>

Device::Device(int device_number):
  device_number_(device_number)
{  
  CHECK_ERROR(cudaSetDeviceFlags(cudaDeviceBlockingSync));
  CHECK_ERROR(cudaSetDevice(device_number_));  
  std::cerr<<"device_number_ = "<<device_number_<<std::endl;

  cudaDeviceProp device_properties;
  CHECK_ERROR(cudaGetDeviceProperties(&device_properties, device_number_));
  total_global_mem_ = device_properties.totalGlobalMem;
  multi_processor_count_ = device_properties.multiProcessorCount;  
}

Device::~Device()
{  
  CHECK_ERROR(cudaDeviceReset());
}

