# photon_propagator
CUDA project that illustrates how photons propagate through Antarctic ice.

## Introduction
This project is meant to be a teaching and learning tool for physicists on IceCube to learn CUDA GPGPU programming using an
example that's relevant to their research.  IceCube currently uses GPUs primarily for the direct propagation of photons in Antarctic ice. The code here is a refactored version of IceCube's internal production project, written in a blend of modern C++14 and CUDA.

# Getting Started
## Dependencies
* [cmake](https://cmake.org/) (version 3.10.2)
* [Boost](https://www.boost.org/) (version 1.65)
* [CUDA](https://developer.nvidia.com/cuda-toolkit-archive) (version 10.1)
Note: The versions above are not strict, but only indicate the versions tested.

### Tested on Ubuntu 18.04
System versions of cmake and boost provided via apt repositories work just fine.  CUDA and GPU driver installations are more involved but can be found [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
```sh
apt-get install cmake libboost-all-dev
```
Note: Currently only boost::filesystem and boost::random are strictly required to build.

## Building
```sh
mkdir build
cd build
cmake <path_to_source>
make
```

## Testing
Running the tests, you should see similar output as below:
```sh
 $ ctest
Test project /home/olivas/photon_propagator/build
    Start 1: test_configuration
1/8 Test #1: test_configuration ...............   Passed    0.12 sec
    Start 2: test_ice_model
2/8 Test #2: test_ice_model ...................   Passed    0.11 sec
    Start 3: test_geometry
3/8 Test #3: test_geometry ....................   Passed    0.16 sec
    Start 4: test_optical_module_lines
4/8 Test #4: test_optical_module_lines ........   Passed    0.05 sec
    Start 5: test_hits
5/8 Test #5: test_hits ........................   Passed    0.11 sec
    Start 6: test_photons
6/8 Test #6: test_photons .....................   Passed    0.11 sec
    Start 7: test_photon_position_direction
7/8 Test #7: test_photon_position_direction ...   Passed    0.12 sec
    Start 8: test_propagation
8/8 Test #8: test_propagation .................   Passed    0.13 sec

100% tests passed, 0 tests failed out of 8

Total Test time (real) =   0.93 sec
```
