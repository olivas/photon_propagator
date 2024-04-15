#include <cassert>

#include <iostream>
#include <vector>
#include <array>

#include <photon_propagator/cpp/device.hpp>
#include <photon_propagator/cpp/test_rotate.hpp>
#include <photon_propagator/cuda/random.cuh>

using std::cerr;
using std::endl;

int main(int argc, char* argv[]){

  if(argc != 1){
    std::cerr<<"Usage: test_random"<<std::endl;
    std::exit(-1);
  }

  std::shared_ptr<Device> device(new Device(0));
  
  const unsigned number_of_blocks{64};
  const unsigned threads_per_block{64};
  std::shared_ptr<Random> rng(new Random(device, number_of_blocks, threads_per_block));
  rng->initialize(41, 0);
  
  std::vector<std::array<float, 3>> result;
  // need to initialize
  const unsigned n_concurrent_threads{number_of_blocks*threads_per_block};
  result.reserve(n_concurrent_threads);
  result.assign(n_concurrent_threads, std::array<float,3>{1., 0., 0.});

  std::cerr<<"input[0][0] = "<<result[0][0]<<std::endl;
  std::cerr<<"input[0][1] = "<<result[0][1]<<std::endl;
  std::cerr<<"input[0][2] = "<<result[0][2]<<std::endl;
  std::cerr<<"input[1][0] = "<<result[1][0]<<std::endl;
  std::cerr<<"input[1][1] = "<<result[1][1]<<std::endl;
  std::cerr<<"input[1][2] = "<<result[1][2]<<std::endl;
  
  float cs{0.1};
  float si{0.1};
  test::test_rotations(result,
		       number_of_blocks,
		       threads_per_block,
		       rng,
		       cs,
		       si);

  unsigned counter{0};
  for(auto v: result){
    std::cerr<<"["<<counter<<"]:("<<v[0]<<","<<v[1]<<","<<v[2]<<")"<<std::endl;
    counter++;
  }
}

  
