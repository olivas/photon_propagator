#pragma once

#include <iostream>

using std::cerr;
using std::endl;

#define CHECK_ERROR(call){ check_error(call, __FILE__, __LINE__);}

inline void check_error(cudaError result, const char* file, int line){
    if(result!=cudaSuccess){
      cerr<<std::string(file)<<"("<<line<<")"<<" CUDA Error ("<<result<<"): "
	  <<cudaGetErrorString(result)<<endl;
      exit(result);
    }
}
