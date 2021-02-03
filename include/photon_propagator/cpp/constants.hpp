#pragma once

static const float ZOFF{1948.07};

const unsigned PHOTONS_PER_THREAD{1024}; // maximum number of photons propagated by one thread
const unsigned PHOTON_BUNCH_SIZE{10};    // size of photon bunches along the muon track

#define TILT         // enable tilted ice layers
#define ANIZ         // enable anisotropic ice

const unsigned N_WAVELENGTH_SLICES{32};       // number of wavelength slices
#define MAXGEO 5200                           // maximum number of OMs
const unsigned MAX_RAND_MULTIPLIERS{131072};  // max. number of random number multipliers

#define XXX 1.e-5f
#define OMR 0.16510f // DOM radius [m]

static unsigned int photon_bunch_size=PHOTON_BUNCH_SIZE;
