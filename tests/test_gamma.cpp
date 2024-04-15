#include <cassert>

#include <iostream>
#include <memory>

#include <photon_propagator/cpp/device.hpp>
#include <photon_propagator/cpp/gamma.hpp>

using std::cerr;
using std::endl;

int main(int argc, char* argv[]){

  std::shared_ptr<Device> device(new Device(0));

  const unsigned number_of_blocks{64};
  const unsigned threads_per_block{64};
  std::shared_ptr<Random> rng(new Random(device, number_of_blocks, threads_per_block));
  rng->initialize(41, 0); // this launches a kernel 
  
  std::vector<float> result;

  Gamma generator(device, rng);
  float k{0.5};
  generator.gamma(result, k, number_of_blocks, threads_per_block);

  std::cerr<<"result.size()="<<result.size()<<std::endl;
  std::cerr<<"result[0] = "<<result[0]<<std::endl;
  std::cerr<<"result[1] = "<<result[1]<<std::endl;
  std::cerr<<"result[2] = "<<result[2]<<std::endl;
  std::cerr<<"result[3] = "<<result[3]<<std::endl;
  std::cerr<<"result[4] = "<<result[4]<<std::endl;
  std::cerr<<"result[5] = "<<result[5]<<std::endl;
  std::cerr<<"result[6] = "<<result[6]<<std::endl;
  std::cerr<<std::endl;      
}

  
