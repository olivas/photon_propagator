#include <cassert>

#include <iostream>

#include <photon_propagator/cpp/device.hpp>
#include <photon_propagator/cpp/random.hpp>

using std::cerr;
using std::endl;

int main(int argc, char* argv[]){

  if(argc != 1){
    std::cerr<<"Usage: test_random"<<std::endl;
    std::exit(-1);
  }

  std::shared_ptr<Device> device(new Device(0));

  //const unsigned number_of_blocks{6};
  //const unsigned threads_per_block{1024};
  //
  // Th grid configuration above causes curand_init
  // to silently fail, leading to unusable
  // states.  The numbers resulting from sebsequent
  // calls are *highly* correlated and each
  // stream sees the same random number sequence.
  // 
  // TODO: Investigate recommendations for grid
  //       configurations as well as the best
  //       way to test that curand_init was
  //       successful.
  
  const unsigned number_of_blocks{64};
  const unsigned threads_per_block{64};
  Random rng(device, number_of_blocks, threads_per_block);
  
  std::vector<unsigned> result_random;
  std::vector<float> result_uniform;
  std::vector<float> result_gamma;
  
  rng.initialize(41, 0);
  rng.random(result_random);
  rng.uniform(result_uniform);  
  rng.gamma(result_gamma, 0.5); 

  std::cerr<<"result_random.size()="<<result_random.size()<<std::endl;
  std::cerr<<"result_random[0] = "<<result_random[0]<<std::endl;
  std::cerr<<"result_random[1] = "<<result_random[1]<<std::endl;
  std::cerr<<"result_random[6] = "<<result_random[6]<<std::endl;
  std::cerr<<std::endl;      

  std::cerr<<"result_uniform.size()="<<result_uniform.size()<<std::endl;
  std::cerr<<"result_uniform[0] = "<<result_uniform[0]<<std::endl;
  std::cerr<<"result_uniform[1] = "<<result_uniform[1]<<std::endl;
  std::cerr<<"result_uniform[6] = "<<result_uniform[6]<<std::endl;
  std::cerr<<std::endl;      

  std::cerr<<"result_gamma.size()="<<result_gamma.size()<<std::endl;
  std::cerr<<"result_gamma[0] = "<<result_gamma[0]<<std::endl;
  std::cerr<<"result_gamma[1] = "<<result_gamma[1]<<std::endl;
  std::cerr<<"result_gamma[6] = "<<result_gamma[6]<<std::endl;
  std::cerr<<std::endl;      
}

  
