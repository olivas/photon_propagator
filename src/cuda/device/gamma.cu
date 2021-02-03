
#include <curand_kernel.h>

__device__ float gamma(float k, curandState_t* state_ptr){
  // gamma distribution
  float x;
  if(k<1){  // Weibull algorithm
    float c=1/k;
    float d=(1-k)*powf(k, 1/(c-1));
    float z;
    float e;
    do{
      z=-logf(curand_uniform(state_ptr));
      e=-logf(curand_uniform(state_ptr));
      x=powf(z, c);
    } while(z+e<d+x);
  }else{  // Cheng's algorithm
    float b=k-logf(4.0f);
    float l=sqrtf(2*k-1);
    float c=1+logf(4.5f);
    float u, v, y, z, r;
    do{
      u=curand_uniform(state_ptr);
      v=curand_uniform(state_ptr);
      y=-logf(1/v-1)/l;
      x=k*expf(y);
      z=u*v*v;
      r=b+(k+l)*y-x;
    } while(r<4.5f*z-c && r<logf(z));
  }
  return x;
}

