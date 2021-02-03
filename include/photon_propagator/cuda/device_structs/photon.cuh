#pragma once

// Position and direction of a photon
// as well as the parent_id of the track
// it originated from.
struct photon{
  float4 r;        // location, time
  float3 n;        // direction
  unsigned int q;  // wavelength slice index
};
