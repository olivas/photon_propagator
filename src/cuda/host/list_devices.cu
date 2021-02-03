
void list_devices(){
  int deviceCount;
  int driver;
  int runtime;
  cudaGetDeviceCount(&deviceCount);
  cudaDriverGetVersion(&driver);
  cudaRuntimeGetVersion(&runtime);
  fprintf(stderr, "Found %d devices, driver %d, runtime %d\n", deviceCount, driver, runtime);
  for(int device=0; device<deviceCount; ++device){
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device);
    fprintf(stderr, "%d(%d.%d): %s %g GHz G(%lu) S(%lu) C(%lu) R(%d) W(%d)\n"
	    "\tl%d o%d c%d h%d i%d m%d a%lu M(%lu) T(%d: %d,%d,%d) G(%d,%d,%d)\n",
	    device, prop.major, prop.minor, prop.name, prop.clockRate/1.e6,
	    (unsigned long)prop.totalGlobalMem, (unsigned long)prop.sharedMemPerBlock,
	    (unsigned long)prop.totalConstMem, prop.regsPerBlock, prop.warpSize,
	    prop.kernelExecTimeoutEnabled, prop.deviceOverlap, prop.computeMode,
	    prop.canMapHostMemory, prop.integrated, prop.multiProcessorCount,
	    (unsigned long)prop.textureAlignment,
	    (unsigned long)prop.memPitch, prop.maxThreadsPerBlock,
	    prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2],
	    prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  }
  fprintf(stderr, "\n");
}

