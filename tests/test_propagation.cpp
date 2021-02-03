#include <cassert>

#include <iostream>
#include <boost/filesystem.hpp>

#include <photon_propagator/device.hpp>
#include <photon_propagator/geometry.hpp>
#include <photon_propagator/configuration.hpp>
#include <photon_propagator/optical_module_lines.hpp>
#include <photon_propagator/tracks.hpp>
#include <photon_propagator/particle.hpp>
#include <photon_propagator/propagator.hpp>

using std::cerr;
using std::endl;

using boost::filesystem::path;
using boost::filesystem::exists;

int main(int argc, char* argv[]){

  if(argc != 3){
    std::cerr<<"Usage: test_geometry <GEOMETRY_FILE> <ICE_MODEL_PATH>"<<std::endl;
    std::exit(-1);
  }

  path geometry_filename(argv[1]);
  assert(exists(path(geometry_filename)));
  
  path ice_model_path(argv[2]);
  assert(exists(path(ice_model_path)));

  std::shared_ptr<Device> device(new Device(0));
  
  Geometry geometry(geometry_filename);
  Configuration config(ice_model_path);
  OpticalModuleLines om_lines(geometry, config);

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

  // These should know how to generate their own photons.
  Tracks tracks(n_tracks, device);
  Cascades cascades(n_cascades, device);
  
  Propagator propagator(device,
			config,
			ice_model,
			geometry,
			om_lines,
			hit_buffer_size,
			photon_buffer_size);
  
  propagator.propagate(photons);
  
}

  
