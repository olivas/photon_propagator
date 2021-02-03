#include <utility>

namespace photon_yield{
  namespace cascade{
    float yield(float E, int type);
    std::pair<float, float> longitudinal_profile_parameters(float E, int type);
  }

  namespace bare_muon{
    float yield(float E, float dr);
    float sub_threshold_cascades_fraction(float E);
  }
};
