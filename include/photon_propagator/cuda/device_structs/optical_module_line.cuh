#pragma once

const unsigned CX{21};   // number of cell columns 
const unsigned CY{19};   // number of cell rows on
const unsigned NSTR{94}; // number of strings
//const unsigned ANUM{11}; // number of coefficients in the angular sensitivity curve

// there are 94 lines.  NSTR == number of strings
// We have 86 strings in total.
// Strings 1-78 are 'normal' strings.
// 79-86 contain HQE DOMs with different spacings.
// sizeof(line) = 32 bytes
// we have 94 of them
// this accounts for 3k
struct optical_module_line{
  short n;    // index of the first dom in the DOM geometry array
  short max;  // number of DOMs on this string
  float x;    // average x coordinate of the string center
  float y;    // average y coordinate of the string center
  float r;    // string radius related to DOM radius account for positional fluctuations and oversize
  float h;    // height of the string.  position of the top DOM.
  float d;    // 1/average_DOM_spacing (1/17m or 1/7m)
  float dl;   // minimum deviation from ideal DOM depth assuming average DOM spacing
  float dh;   // maximum deviation from ideal DOM depth assuming average DOM spacing

  optical_module_line():
    n(0), max(0),    
    x(0.), y(0.), r(0.),
    d(0.), dl(0.), dh(0.)
  {};
};
