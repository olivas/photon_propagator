#pragma once

struct hit{
  unsigned int i; // index into q (DOM index?)
  float t;        // i'm guessing time...yes.  r.w fourth vector component.
  float z;        // i'm guessing z-position...nope.
                  // it's wavelength, of course.
};
