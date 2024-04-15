#include <curand_kernel.h>

#include <photon_propagator/cuda/rotate.cuh>

__device__
void rotate(const float cs, // cosine
            const float si, // sine
            float3& n,      // direction
            curandState_t* state_ptr){
    float3 p1;
    float3 p2;
    int i=0;
    {
        float3 r;
        r.x = n.x*n.x;
        r.y = n.y*n.y;
        r.z = n.z*n.z;
        if(r.y>r.z){
            if(r.y>r.x)
                i = (swap(n.x,n.y), swap(r.x,r.y), 1);
        }
        else{
            if(r.z>r.x)
                i = (swap(n.x,n.z), swap(r.x,r.z), 2);
        }

        r.y = rsqrtf(r.x+r.y);
        p1.x = -n.y*r.y;
        p1.y = n.x*r.y;
        p1.z = 0;

        r.z = rsqrtf(r.x+r.z);
        p2.x = -n.z*r.z;
        p2.y = 0;
        p2.z = n.x*r.z;
    }

    {
        float4 q1;

        q1.x = p1.x-p2.x;
        q1.y = p1.y-p2.y;
        q1.z = p1.z-p2.z;
        p2.x += p1.x;
        p2.y += p1.y;
        p2.z += p1.z;

        q1.w = rsqrtf(q1.x*q1.x + q1.y*q1.y + q1.z*q1.z);
        p1.x = q1.x*q1.w;
        p1.y = q1.y*q1.w;
        p1.z = q1.z*q1.w;

        q1.w = rsqrtf(p2.x*p2.x + p2.y*p2.y + p2.z*p2.z);
        p2.x *= q1.w;
        p2.y *= q1.w;
        p2.z *= q1.w;
    }

    {
        float2 p;
        float xi = 2*FPI*curand_uniform(state_ptr);
        sincosf(xi, &p.y, &p.x);

        n.x = cs*n.x + si*(p.x*p1.x + p.y*p2.x);
        n.y = cs*n.y + si*(p.x*p1.y+ p.y*p2.y);
        n.z = cs*n.z + si*(p.x*p1.z + p.y*p2.z);

        float r = rsqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
        n.x*=r;
        n.y*=r;
        n.z*=r;
        if(i==1)
            swap(n.x,n.y);
        else
        if(i==2)
            swap(n.x,n.z);
    }
}


__global__
void test_kernel_rotate(const float cs,       // cosine
			const float si,       // sine
			curandState_t* state, // rng
			float3* result){      // direction
			
  int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
  /* Copy state to local memory for efficiency */
  curandState local_state = state[thread_idx];
  /* Scramble the input vector */
  rotate(cs, si, result[thread_idx], &local_state);
  /* Copy state back to global memory */
  state[thread_idx] = local_state;
}
