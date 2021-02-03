#pragma once

const unsigned MAX_ICE_LAYERS{180}; // maximum number of ice layers
struct ice_color{
  float wvl;     // wavelength of this block
  float ocm;     // 1 / speed of light in medium
  float coschr;  // cos and sin of the cherenkov angle
  float sinchr;  // cos and sin of the cherenkov angle

  struct{
    float abs; // absorption
    float sca; // scattering
  } ice_properties [MAX_ICE_LAYERS];
};
