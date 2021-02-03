
add_library(photon_propagator SHARED
  src/cpp/photon_yield.cpp
  )

cuda_add_library(photon_propagator_cuda SHARED
  src/cuda/device/gamma.cu
  src/cuda/device.cu
  src/cuda/random.cu
  #src/cuda/cascades_to_photons.cu 
  #src/cuda/configuration.cu
  #src/cuda/ice_model.cu
  #src/cuda/geometry.cu
  #src/cuda/optical_module_lines.cu
  #src/cuda/hits.cu
  #src/cuda/photons.cu
  #src/cuda/tracks.cu
  #src/cuda/cascades.cu
  #src/cuda/rotate.cu
  #src/cuda/tracks_to_photons.cu
  #src/cuda/propagator.cu
  )

# cascade_to_photons and random are compiled into two separate
# object files, leading to multiple definition linking errors.
set_target_properties(photon_propagator_cuda
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(photon_propagator_cuda
  PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(photon_propagator_cuda photon_propagator)
