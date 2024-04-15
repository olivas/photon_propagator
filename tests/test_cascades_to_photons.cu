#include <cassert>

#include <iostream>
#include <boost/filesystem.hpp>

#include <photon_propagator/cpp/device.hpp>
#include <photon_propagator/cpp/geometry.hpp>
#include <photon_propagator/cpp/configuration.hpp>
#include <photon_propagator/cpp/ice_model.hpp>
#include <photon_propagator/cpp/particle.hpp>
#include <photon_propagator/cpp/cascades.hpp>
#include <photon_propagator/cpp/photons.hpp>
#include <photon_propagator/cpp/cascades_to_photons.hpp>
#include <photon_propagator/cuda/random.cuh>

using std::cerr;
using std::endl;

using boost::filesystem::path;
using boost::filesystem::exists;

int main(int argc, char* argv[]){

  if(argc != 2){
    std::cerr<<"Usage: test_cascades_to_photons <ICE_MODEL_PATH>"<<std::endl;
    std::exit(-1);
  }
  
  path ice_model_path(argv[1]);
  assert(exists(path(ice_model_path)));

  std::shared_ptr<Device> device(new Device(0));
  
  Configuration config(ice_model_path);
  float scattering_angle{config.get_configuration().g};  
  IceModel ice_model(ice_model_path, scattering_angle);

  size_t photon_buffer_size{1000};
  size_t hit_buffer_size{100};

  particle electron;
  electron.major_id=0;
  electron.minor_id=0;
  electron.ptype=particle_type::e_minus;
  electron.time=0.;
  electron.energy=100.; // in god-given units
  electron.length=0.;  
  electron.direction={0,0,0};
  electron.position={0,0,0};

  Cascades cascades(1, device);
  cascades.add(electron);
  
  unsigned number_of_blocks{64};
  unsigned threads_per_block{64};
  unsigned n_concurrent_threads{number_of_blocks*threads_per_block};
  Photons photons(n_concurrent_threads, device);
  Random rng(device, number_of_blocks, threads_per_block);
  cascades::GeneratePhotons(rng,
			    ice_model,
			    cascades,
			    device,
			    number_of_blocks,
			    threads_per_block,
			    photons);
  
  photons.pprint();
}

  
