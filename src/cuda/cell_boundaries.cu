#include <iostream>
#include <sstream>
#include <fstream>
#include <numeric>

#include <boost/filesystem.hpp>

#include <photon_propagator/optical_module_lines.hpp>
#include <photon_propagator/cuda/check_error.cuh>

using std::ifstream;
using std::stringstream;
using std::string;
using std::cerr;
using std::endl;
using std::getline;
using std::map;

const unsigned CX{21};   // number of cell columns 
const unsigned CY{19};   // number of cell rows on

CellBoundaries::CellBoundaries(const Geometry& geometry,
			       const Configuration& configuration,
			       const OpticalModuleLines& optical_module_lines){
  configure(geometry, configuration, optical_module_lines);
};

void
CellBoundaries::configure(const Geometry& geometry,
			  const Configuration& configuration,
			  const OpticalModuleLines& optical_module_lines){
  
  float[2][2] cb = configuration.get_cb();  
  std::vector<float> magic_0;
  std::vector<float> magic_1;
  for(auto p: optical_module_lines.get_optical_module_lines()){
    const optical_module_line& om_line = p.second;
  
    float magic_number_0 = cb[0][0]*om_line.x + cb[1][0]*om_line.y;
    float magic_number_1 = cb[0][1]*om_line.x + cb[1][1]*om_line.y;

    magic_0.push_back(magic_number_0);
    magic_1.push_back(magic_number_1);    
  }

  // cell lower boundary (low)
  // i think these are cell dimensions
  float cl[2] = {
    *std::min_element(begin(magic_0), end(magic_0)),
    *std::min_element(begin(magic_1), end(magic_1))
  };
  
  // cell upper boundary (high)
  float ch[2] = {
    *std::max_element(begin(magic_0), end(magic_0)),
    *std::max_element(begin(magic_1), end(magic_1))
  };


  float crst[2] = { 
    (CX-1)/(ch[0]-cl[0]),
    (CY-1)/(ch[1]-cl[1]),
  };

  // Need to figure out exactly what role each plays:
  //  1) cb  (hint: it's a 2x2 matrix)
  //  2) DIR1 and DIR2 (i think these are string planes)
  //        9.3 and 129.3 (not orthogonal, but neither are our string planes)
  //  The rest should fall into place once these are grokked.
    
  map<unsigned char, int> cells[CX][CY];
  for(auto p: optical_module_lines.get_optical_module_lines()){
    const unsigned string_number = p.first;
    const optical_module_line& om_line = p.second;
    
  }
  
  for(map<unsigned char, short>::iterator i=num.begin(); i!=num.end(); ++i){
    line& s = d.sc[i->second];
    
    // n has to be the cell index
    // and we want to know how many strings
    // are in each cell
    float magic_number_0 = cb[0][0]*om_line.x + cb[1][0]*om_line.y;
    float magic_number_1 = cb[0][1]*om_line.x + cb[1][1]*om_line.y;
      
    //if(n[m]<0 || n[m] >= d.cn[m]){ 
    //  cerr<<"Error in cell initialization"<<endl; 
    //  exit(1); 
    //}

    int n0 = lroundf((magic_number_0 - cl[0]) * crst[0]);
    int n1 = lroundf((magic_number_1 - cl[1]) * crst[1]);
    
    float d1_0 = fabsf(magic_number_0 - (cl[0] + (n0 - 0.5f)/crst[0]));
    float d1_1 = fabsf(magic_number_1 - (cl[1] + (n1 - 0.5f)/crst[1]));

    float d2_0 = fabsf(magic_number_0 - (cl[0] + (n0 + 0.5f)/crst[0]));
    float d2_1 = fabsf(magic_number_1 - (cl[1] + (n1 + 0.5f)/crst[1]));

    float d = min(d1, d2) * sin12-s.r;
    //if(d<0){ 
    //  flag=false; 
    //  cerr<<"Warning: string "<<(int)i->first<<" too close to cell boundary"<<endl; 
    //}
    //
    // cells is an array of maps
    // each cell contains a map<key = string_number, value = count>
    cells[n0][n1][i->first]++;
  }

//  for(int m=0; m<2; m++){ 
//    d.cl[m]=cl[m]; 
//    d.crst[m]=crst[m]; 
//  }

  
//  for(int m=0; m<2; m++){ 
//    d.cl[m]=cl[m]; 
//    d.crst[m]=crst[m]; 
//  }

}  

void CellBoundaries::to_device(){

//  unsigned long sizeof_om_lines{om_lines_.size()*sizeof(optical_module_line)};
//  std::cerr<<"allocated "<<sizeof_om_lines
//	   <<" bytes ("<<int(sizeof_om_lines/1e3)<<" kB) for optical module lines.\n";
//  CHECK_ERROR(cudaMalloc((void**) &__device_ptr, sizeof_om_lines));
//  CHECK_ERROR(cudaMemcpy(__device_ptr, om_lines_.data(), sizeof_om_lines, cudaMemcpyHostToDevice));  
}