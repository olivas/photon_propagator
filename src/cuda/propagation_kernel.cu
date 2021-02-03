#include <curand_kernel.h>

#include "util.cu"
#include "gamma.cu"
#include "rotate.cu"

__global__ void propagate(configuration* config,			  
			  curandState_t* rng_state,
			  ices* ice_model,
			  DOM* oms,
			  photon* photons,
			  unsigned int photon_buffer_size, // number of photons to propagate
			  hit* hits,
			  unsigned int hit_buffer_size){			  

  // Q1: What's the difference between pbuf and photon?
  //  A: These have been renamed.
  //     pbuf is now a photon.  (position, direction)
  //     photon is now a track. (position, direction, longitudinal parameters)
  //
  // Q2: Where does each member live? i.e. what type of memory?  All global?
  //
  // Q3: What's the optimal hit buffer size?
  //
  // Q4: How is hit buffer overflow handled?
  // 

  const unsigned thread_index = blockIdx.x + threadIdx.x * gridDim.x;
  /* Copy state to local memory for efficiency */
  curandState_t local_rng_state = rng_state[thread_index];
  curandState_t* state_ptr = &local_rng_state;
  
  unsigned& hit_index = e.hidx;

  { // TODO: grok me
    if(threadIdx.x == 0){
      e=*ed;
      hit_index = gridDim.x * blockDim.x + blockIdx.x;
    }
    __syncthreads();
    
    int old;
    float TOT=0;
  }
  
  // this block initializes photons and spreads them
  // in space depending on whether they're from
  // muon tracks or cascades.  the spread is determined
  // from various parameters:
  //   l,f for tracks.
  //   a,b for cascades.

  // maybe split into multiple kernels?
  unsigned grid_stride = gridDim.x * blockDim.x;
  for(unsigned int i=thread_index;
      i<photon_buffer_size;
      i += grid_stride){
    
      track blob = tracks[i/PHOTON_BUNCH_SIZE];
      float4 r=blob.r; // track position
      float3 n=blob.n; // track direction
      
      float& l=blob.l;
      if(l>0){
	// spread photon over the track length.
	// this allows us to pass chunks of muons
	// does this cost more than it saves?
	l*=curand_uniform(state_ptr);
      }else{
	if(blob.b>0){
	  l=blob.b*gamma(blob.a, state_ptr);
	}
      }
      
      if(l>0){
	r.w+=e.ocv*l;
	r.x+=n.x*l;
	r.y+=n.y*l;
	r.z+=n.z*l;
      }

      if(blob.f<curand_uniform(state_ptr)){
	const float a=0.39f, b=2.61f;
	const float I=1-expf(-b*exp2f(a));
	float cs=max(1-powf(-logf(1-curand_uniform(state_ptr)*I)/b, 1/a), -1.0f);
	float si=sqrtf(1-cs*cs);
	rotate(cs, si, n, state_ptr);
      }

      // j == wavelength slice
      // different ice layers have different indices of refraction
      // and therefore different cherenkov cones.
      unsigned int j=min(__float2int_rd(N_WAVELENGTH_SLICES*curand_uniform(state_ptr)), N_WAVELENGTH_SLICES-1);
      ices* w = &ice_model[j];      
      rotate(w->coschr, w->sinchr, n, state_ptr);

      // Create a photon and add it to the buffer.
      photon ph;
      ph.r=r;
      ph.n=n;
      ph.q=j;
      photon_buffer[i]=ph;
  }
  __threadfence_block();


  // Now that photon_buffer is filled
  // propagate them.
  float sca; // scattering length
  unsigned int niw=0;  
  for(unsigned int i=thread_index;
      i<photon_buffer_size;
      TOT==0 && (i=atomicAdd(&hit_index, gridDim.x)) ){
    // this 'increment' statement is a little evil
    //   TOT==0 && (i=atomicAdd(&hit_index, gridDim.x)
    // it increments hit_index by the gridDim.x only when TOT is 0.
    // remember the hit_index is a reference to the global hidx
    // in the global 'dats' (i.e. context) struct.

    // this code is sooo fucked up.
    int om=-1;
    float3 n={0,0,0};   // photon direction
    float4 r={0,0,0,0}; // photon position
    ices* w;
    if(TOT==0){ // initialize photon
      photon f=photon_buffer[i];
      r=f.r;
      n=f.n;
      w = &ice_model[f.q];
      niw = photons[i/PHOTON_BUNCH_SIZE].q;
      om = -1;
      TOT = -logf(curand_uniform(state_ptr));
      sca=0;
    }

    // if there is any time TOT is non-zero
    // sca is not initialized and this is undefined behavior.
    // it will randomly and almost always evaluate to false.
    if(sca==0){ // get distance for overburden
      float SCA = -logf(curand_uniform(state_ptr));
      old=om;
      float tot;
      float z=r.z;
      float nr=1.f;

      int i=__float2int_rn((z-e.hmin)*e.rdh);
      if(i<0) i=0; else if(i>=e.size) i=e.size-1;
      float h=e.hmin+i*e.dh; // middle of the layer
      h=n.z<0?h-e.hdh:h+e.hdh;

      float ais=(n.z*SCA-(h-z)*w->ice_properties[i].sca)*e.rdh;
      float aia=(n.z*TOT-(h-z)*w->ice_properties[i].abs)*e.rdh;

      int j=i;
      if(n.z<0) for(; j>0 && ais<0 && aia<0; h-=e.dh, ais+=w->ice_properties[j].sca, aia+=w->ice_properties[j].abs) --j;
      else for(; j<e.size-1 && ais>0 && aia>0; h+=e.dh, ais-=w->ice_properties[j].sca, aia-=w->ice_properties[j].abs) ++j;

      if(i==j || fabsf(n.z)<XXX) sca=SCA/w->ice_properties[j].sca, tot=TOT/w->ice_properties[j].abs;
      else sca=(ais*e.dh/w->ice_properties[j].sca+h-z)/n.z, tot=(aia*e.dh/w->ice_properties[j].abs+h-z)/n.z;

      // get overburden for distance
      if(tot<sca){
	sca=tot;
	TOT=0;
      }else{
	TOT=nr*(tot-sca)*w->ice_properties[j].abs;
      }
    }

    om=-1;
    float del=sca;      // W.
    { // sphere         // T.
      float & sca = del;// F. ???
      float2 ri, rf, pi, pf;

      ri.x=r.x;
      rf.x=r.x+sca*n.x;
      
      ri.y=r.y;
      rf.y=r.y+sca*n.y;

      ctr(e, ri, pi);
      ctr(e, rf, pf);

      ri.x=min(pi.x, pf.x)-e.rx; rf.x=max(pi.x, pf.x)+e.rx;
      ri.y=min(pi.y, pf.y)-e.rx; rf.y=max(pi.y, pf.y)+e.rx;

      int2 xl, xh;

      xl.x=min(max(__float2int_rn((ri.x-e.cl[0])*e.crst[0]), 0), e.cn[0]);
      xh.x=max(min(__float2int_rn((rf.x-e.cl[0])*e.crst[0]), e.cn[0]-1), -1);

      xl.y=min(max(__float2int_rn((ri.y-e.cl[1])*e.crst[1]), 0), e.cn[1]);
      xh.y=max(min(__float2int_rn((rf.y-e.cl[1])*e.crst[1]), e.cn[1]-1), -1);

      // i'm thinking we just calculated cell boundaries
      
      for(int i=xl.x, j=xl.y;
	  i<=xh.x && j<=xh.y;
	  ++j<=xh.y?:(j=xl.y, i++))
	for(unsigned char k=e.is[i][j]; k!=0x80; ){
	  unsigned char m=e.ls[k];
	  line & s = e.sc[m&0x7f];
	  k=m & 0x80 ? 0x80:k+1;

	  float b=0, c=0, dr;
	  dr=s.x-r.x;
	  b+=n.x*dr; c+=dr*dr;
	  dr=s.y-r.y;
	  b+=n.y*dr; c+=dr*dr;

	  float np=1-n.z*n.z;
	  float D=b*b-(c-s.r*s.r)*np;
	  if(D>=0){
	    D=sqrtf(D);
	    float h1=b-D, h2=b+D;
	    if(h2>=0 && h1<=sca*np){
	      if(np>XXX){
		h1/=np;
		h2/=np;
		if(h1<0) h1=0;
		if(h2>sca) h2=sca;
	      }else {
		h1=0;
		h2=sca;
	      }
	      
	      h1=r.z+n.z*h1, h2=r.z+n.z*h2;
	      float zl, zh;
	      if(n.z>0){
		zl=h1;
		zh=h2;
	      }else{
		zl=h2;
		zh=h1;
	      }

	      int omin=0, omax=s.max;
	      int n1=s.n-omin+min(omax+1, max(omin, __float2int_ru(omin-(zh-s.dl-s.h)*s.d)));
	      int n2=s.n-omin+max(omin-1, min(omax, __float2int_rd(omin-(zl-s.dh-s.h)*s.d)));
	      
	      for(int l=n1; l<=n2; l++) if(l!=old){
		  if(l==-1) 
		    continue;
		  const DOM & dom=oms[l];
		  float b=0, c=0, dr;
		  dr=dom.r[0]-r.x;
		  b+=n.x*dr; c+=dr*dr;
		  dr=dom.r[1]-r.y;
		  b+=n.y*dr; c+=dr*dr;
		  dr=dom.r[2]-r.z;
		  b+=n.z*dr; c+=dr*dr;
		  float D=b*b-c+e.R2;
		  if(D>=0){
		    float sqd=sqrtf(D);
		    float h=b-sqd*e.zR;
		    if(h>0 && h<=del) om=l, del=h;
		  }
		}
	    }
	  }
	}
    }
    
    sca-=del;
    { // advance
      r.x+=del*n.x;
      r.y+=del*n.y;
      r.z+=del*n.z;
      r.w+=del*w->ocm;
    }

    if(!isfinite(TOT) || !isfinite(sca)){
      TOT=0;
      om=-1;
    }

    float xi=curand_uniform(state_ptr);
    if(om!=-1){
      // this means we have a hit!
      // create a hit object and add
      // it to the hit buffer.
      
      hit h;
      h.i=om;
      h.t=r.w;
      h.n=niw;
      h.z=w->wvl;

      float sum;
      {
	float& x = n.z;
	float y=1;
	sum=e.s[0];
	for(int i=1; i<ANUM; i++){
	  y*=x;
	  sum+=e.s[i]*y;
	}
      }

      if(e.mas*xi<sum){
	unsigned int j = atomicAdd(&ed->hidx, 1);
	if(j<e.hit_buffer_size)
	  hits[j]=h;
	  // and if it's greater?
	  // we just gloss over that?
      }

      if(e.zR==1) TOT=0; else old=om;
    }
    else if(TOT<XXX) TOT=0;
    else{
      float &sf=e.sf, &g=e.g, &g2=e.g2, &gr=e.gr;

      if(xi>sf){
	xi=(1-xi)/(1-sf);
	xi=2*xi-1;
	if(g!=0){
	  float ga=(1-g2)/(1+g*xi);
	  xi=(1+g2-ga*ga)/(2*g);
	}
      }
      else{
	xi/=sf;
	xi=2*powf(xi, gr)-1;
      }

      if(xi>1) xi=1; else if(xi<-1) xi=-1;

      float si=sqrtf(1-xi*xi);
      rotate(xi, si, n, state_ptr);

    } // what does this close?
  } // what does this close?
  // This is the end of the photon propagation loop
  //  for(unsigned int i=thread_index;
  //      i<photon_buffer_size;
  //      TOT==0 && (i=atomicAdd(&hit_index, gridDim.x)) ){

  /* Copy state back to global memory */
  rng_state[thread_index] = local_rng_state;
  
  __syncthreads();
  __threadfence();
}
