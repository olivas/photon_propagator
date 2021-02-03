#pragma once

const unsigned ANUM{11}; // number of coefficients in the angular sensitivity curve

struct configuration{
  
  float sf;      // scattering function: 0=HG; 1=SAM
  float g;       // g=<cos(scattering angle)>
  float g2;      // g2=g*g
  float gr;      // gr=(1-g)/(1+g)
  float R;       // DOM radius
  float R2;      // radius^2
  float zR;      // inverse "oversize" scaling factor
  float eff;     // OM efficiency correction
  float mas;     // maximum angular sensitivity
  float s[ANUM]; // ang. sens. coefficients  
};
