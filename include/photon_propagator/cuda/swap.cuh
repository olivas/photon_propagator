#pragma once

__device__ void swap(float& x, float& y){
  float a=x;
  x=y;
  y=a;
}
