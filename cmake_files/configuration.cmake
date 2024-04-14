
project(icecube_photon_propagator LANGUAGES CXX CUDA)

# nvcc only currently supports up to C++14 
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 23)

set(CMAKE_BUILD_TYPE DEBUG)

find_package(CUDA REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)

include_directories(include)
include_directories(${Boost_INCLUDE_DIRS})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Can't use gcc beyond 8 since nvcc chokes.
# Note this was only an issue when trying for separate compilation.

# https://devblogs.nvidia.com/separate-compilation-linking-cuda-device-code/
# --device-c allows for relocatable code, but could hurt performance.
# NB: requires at least sm_20 architecture.

# https://devblogs.nvidia.com/building-cuda-applications-cmake/
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
#set(CUDA_NVCC_FLAGS 
#  ${CUDA_NVCC_FLAGS}
#  -Xptxas=-v 
#  -O2
#  -G
#  -g
#  --device-c
#  --use_fast_math 
#  --compiler-options=-O2,--fast-math
#  -D_FORCE_INLINES
#  -Xcompiler -fPIC
#  )
