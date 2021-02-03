#pragma once

// This represents track lengths
// This should be renamed to track
struct cascade{
  unsigned int q;  // ??? no idea...we're gonna use this for n_photons.
  float4 r;        // location, time
  float3 n;        // direction
  float a, b;      // longitudinal development parametrization coefficients
};
