#include <dataclasses/physics/I3Particle.h>
namespace{
  int to_gpu_type(const I3Particle& p){
    if(p.GetLocationType()!=I3Particle::InIce) 
      return -4;
    else if(p.GetShape()==I3Particle::Dark) 
      return -3;
    else{
      switch(p.GetType()){
      default:
        return -2;
      case I3Particle::MuPlus:
      case I3Particle::MuMinus:
        return -1;
      case I3Particle::DeltaE:
      case I3Particle::Brems:
      case I3Particle::PairProd:
        return 1;
      case I3Particle::EMinus:
        return 2;
      case I3Particle::EPlus:
        return 3;
      case I3Particle::Gamma:
        return 4;
      case I3Particle::NuclInt:
      case I3Particle::Hadrons:
        return 101;
      case I3Particle::PiPlus:
        return 102;
      case I3Particle::PiMinus:
        return 103;
      case I3Particle::K0_Long:
        return 104;
      case I3Particle::PPlus:
        return 105;
      case I3Particle::Neutron:
        return 106;
      case I3Particle::PMinus:
        return 107;
      }
    }
  }
}
