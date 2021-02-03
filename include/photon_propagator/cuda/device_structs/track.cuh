#pragma once

// This represents track lengths
// This should be renamed to track
struct track{
  unsigned int q;  // track id
  float4 r;        // location, time
  float3 n;        // direction
  float l;         // track length
  float f;         // fraction of light from muon alone (without cascades)
};
