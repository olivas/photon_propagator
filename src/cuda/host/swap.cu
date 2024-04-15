//
// Created by olivas on 4/14/24.
//
#include <photon_propagator/cuda/swap.cuh>

__device__ void swap(float& x, float& y){
    float a=x;
    x=y;
    y=a;
}
