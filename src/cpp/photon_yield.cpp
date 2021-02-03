
#include <algorithm>
#include <cmath>
#include <cassert>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

#include <photon_propagator/cpp/photon_yield.hpp>

namespace{
  const float PPM{2450.08};     // photons per meter
}

float
photon_yield::cascade::yield(float E, int type){
    
  /**
   * a,b describe the longitudinal profile of cascades.
   * The energy dependence of a is given by p1+p2*logE and b is constant.
   * Add comment about validity of parameterization for low-energy cascades.
   *
   * Total light yield of cascades:
   * For e-,e+,gamma the light yield is given by p*simulated_density/ice_density.
   * For hadrons the light yield is scaled down by Fi where i denotes the particle type.
   * The parameterizations have been derived from fits in an energy range
   * from 30 GeV to 10 TeV. Below approximately 10 GeV the F is not described
   * by F = 1-(E/E0)^(-m)*(1-f0) and is therefore a rather crude approximation.
   * Fluctuations have been parameterized with sigma/F = rms0*ln(E[GeV])^(-gamma).
   * For antiprotons the annihilation cross section has to be taken into account
   * at small energies. At the moment F for protons is used but with the fluctuations 
   * of antiprotons.
   *
   * Reference: icecube/201210001 (for hadrons the values are recalculated)
   * The type are as following:
   * type  particle
   * 1 standard em (same as e-) is default if no other type is given
   * 2 e-
   * 3 e+
   * 4 gamma
   * 101 standard hadron (same as pi+)
   * 102 pi+
   * 103 pi-
   * 104 kaon0L
   * 105 proton
   * 106 neutron
   * 107 anti_proton Use F for protons because parameterization fails for small
   *     energies due to annihilation cross section. However sigma_F is used from
   *     anti_proton parameterization.
   **/
    
  float f=1.0f;
  if(type>100){
    float E0, m, f0, rms0, gamma;
    
    switch(type){
    default:
    case 101: // standard hadron (same as pi+)
    case 102: // pi+
      E0=0.18791678f;
      m=0.16267529f;
      f0=0.30974123f;
      rms0 =0.95899551f;
      gamma=1.35589541f;
      break;

    case 103: // pi-
      E0=0.19826506f;
      m=0.16218006f;
      f0=0.31859323f;
      rms0 =0.94033488f;
      gamma=1.35070162f;
      break;

    case 104: // kaon0L
      E0=0.21687243f;
      m=0.16861530f;
      f0=0.27724987f;
      rms0 =1.00318874f;
      gamma=1.37528605f;
      break;

    case 105: // proton
      E0=0.29579368f;
      m=0.19373018f;
      f0=0.02455403f;
      rms0 =1.01619344f;
      gamma=1.45477346f;
      break;

    case 106: // neutron
      E0=0.66725124f;
      m=0.19263595f;
      f0=0.17559033f;
      rms0 =1.01414337f;
      gamma=1.45086895f;
      break;

    case 107: // anti_proton
      E0=0.29579368f;
      m=0.19373018f;
      f0=0.02455403f;
      rms0 =1.01094637f;
      gamma=1.50438415f;
      break;
    }
    
    float e=std::max(2.71828183f, E);
    float F=1-std::pow(e/E0, -m)*(1-f0);
    float dF=F*rms0*std::pow(std::log(e), -gamma);

    // TODO: Check that this produces the same sampling
    //       as was in ppc.
    boost::random::mt19937 rng;
    boost::random::normal_distribution<> gaussian;
    
    do f=F+dF*gaussian(rng);
    while(f<0 || 1.1<f);
  }
  
  const float RHO{0.9216};         // density of ice [mwe]
  const float EM=5.321*0.910f/RHO; // 0.910 density used in simulation
  return PPM*f*EM*E;
}

  
std::pair<float, float>
photon_yield::cascade::longitudinal_profile_parameters(float E, int type){  
  const float M0{0.105658389};  // muon rest mass [GeV]
  const float RHO{0.9216};      // density of ice [mwe]
  const float LRAD=0.39652*0.910f/RHO;

  float logE=std::log(std::max(M0, type<0?10:E));
    
  /**
   * a,b describe the longitudinal profile of cascades.
   * The energy dependence of a is given by p1+p2*logE and b is constant.
   * Add comment about validity of parameterization for low-energy cascades.
   *
   * Total light yield of cascades:
   * For e-,e+,gamma the light yield is given by p*simulated_density/ice_density.
   * For hadrons the light yield is scaled down by Fi where i denotes the particle type.
   * The parameterizations have been derived from fits in an energy range
   * from 30 GeV to 10 TeV. Below approximately 10 GeV the F is not described
   * by F = 1-(E/E0)^(-m)*(1-f0) and is therefore a rather crude approximation.
   * Fluctuations have been parameterized with sigma/F = rms0*ln(E[GeV])^(-gamma).
   * For antiprotons the annihilation cross section has to be taken into account
   * at small energies. At the moment F for protons is used but with the fluctuations 
   * of antiprotons.
   *
   * Reference: icecube/201210001 (for hadrons the values are recalculated)
   * The type are as following:
   * type  particle
   * 1 standard em (same as e-) is default if no other type is given
   * 2 e-
   * 3 e+
   * 4 gamma
   * 101 standard hadron (same as pi+)
   * 102 pi+
   * 103 pi-
   * 104 kaon0L
   * 105 proton
   * 106 neutron
   * 107 anti_proton Use F for protons because parameterization fails for small
   *     energies due to annihilation cross section. However sigma_F is used from
   *     anti_proton parameterization.
   **/

  std::pair<float, float> result(0.,0.);
  switch(type){
  default:
  case 101: // standard hadron (same as pi+)
  case 102: // pi+
    result.first = 1.58357292f+0.41886807f*logE;
    result.second = LRAD/0.33833116f;
    break;
    
  case 103: // pi-
    result.first = 1.69176636f+0.40803489f*logE;
    result.second = LRAD/0.34108075f;
    break;
    
  case 104: // kaon0L
    result.first = 1.95948974f+0.34934666f*logE;
    result.second = LRAD/0.34535151f;
    break;
    
  case 105: // proton
    result.first = 1.47495778f+0.40450398f*logE;
    result.second = LRAD/0.35226706f;
    break;
    
  case 106: // neutron
    result.first = 1.57739060f+0.40631102f*logE;
    result.second = LRAD/0.35269455f;
    break;
    
  case 107: // anti_proton
    result.first = 1.92249171f+0.33701751f*logE;
    result.second = LRAD/0.34969748f;
    break;
  case 1:   // em shower
  case 2:   // e-
    result.first = 2.01849f+0.63176f*logE;
    result.second = LRAD/0.63207f;
    break;
    
  case 3:   // e+
    result.first = 2.00035f+0.63190f*logE;
    result.second = LRAD/0.63008f;
    break;
    
  case 4:   // gamma
    result.first = 2.83923f+0.58209f*logE;
    result.second = LRAD/0.64526f;
    break;
  }
  return result;
}

float photon_yield::bare_muon::sub_threshold_cascades_fraction(float E){  // bare muon
  // cascades below 500MeV
  const float M0{0.105658389};  // muon rest mass [GeV]
  float logE=std::log(std::max(M0, E));
  return 1+std::max(0.0f, 0.1880f+0.0206f*logE); 
}

float photon_yield::bare_muon::yield(float E, float track_length){  // bare muon
  assert(track_length>=0);
  return PPM*track_length*sub_threshold_cascades_fraction(E);
}
