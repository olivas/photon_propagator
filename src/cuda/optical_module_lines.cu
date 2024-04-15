#include <iostream>
#include <sstream>
#include <fstream>
#include <numeric>

#include <boost/filesystem.hpp>

#include <photon_propagator/cpp/optical_module_lines.hpp>
#include <photon_propagator/cuda/check_error.cuh>

using std::ifstream;
using std::stringstream;
using std::string;
using std::cerr;
using std::endl;
using std::getline;
using std::map;

const float DIR1{9.3};
const float DIR2{129.3};
const float FPI{3.141592653589793};

OpticalModuleLines::OpticalModuleLines(const Geometry& geometry,
				       const Configuration& configuration):
  ls_(NSTR, ' ')
{
  calculate_sin12();
  calculate_cb();
  configure(geometry, configuration);
  configure_cells(geometry, configuration);
};

void
OpticalModuleLines::configure(const Geometry& geometry, const Configuration& configuration){

  // let's construct a more convenient geometry
  // this geo has 94 strings (deep core mapped to different string numbers)
  // the key is the string number
  // the positions are ordered by decreasing depth

  const float DOM_RADIUS = configuration.get_configuration().R;
  
  const Geometry::packed_map_t& geo = geometry.get_packed_geometry();  
  for(auto p: geo){
    auto string_number = p.first;
    auto positions = p.second;

    optical_module_line om_line;
    auto sum_x = [](auto init, Geometry::Position p){
      return init + p.x;
    };
    auto sum_y = [](auto init, Geometry::Position p){
      return init + p.y;
    };
    
    om_line.x = std::accumulate(begin(positions), end(positions), 0., sum_x);
    om_line.y = std::accumulate(begin(positions), end(positions), 0., sum_y);
    om_line.x /= positions.size();
    om_line.y /= positions.size();

    // geometry index of top of string
    om_line.n = positions.front().index;

    // string depth
    om_line.h = positions.front().z;

    // average line spacing
    float line_length = positions.front().z - positions.back().z;
    om_line.d = (positions.size()-1)/line_length;

    // set om_line.r to the largest radial difference
    // between the DOM position and the average
    // string center
    std::vector<float> rsquared;
    for(auto position: positions){
      float dx = position.x - om_line.x;
      float dy = position.y - om_line.y;
      rsquared.push_back(dx*dx + dy*dy);
    }
    float max_radius = *(std::max_element(begin(rsquared), end(rsquared)));
    //std::cerr<<"DOM_RADIUS = "<<DOM_RADIUS<<std::endl;
    //std::cerr<<"sqrt(max_radius) = "<<std::sqrt(max_radius)<<std::endl;
    om_line.r = DOM_RADIUS + std::sqrt(max_radius);
    
    // setting h and l
    std::vector<float> dzs;
    for(auto position: positions){
      // position on the string
      unsigned dom_number = position.index - om_line.n;
      float dz = position.z - (om_line.h - dom_number/om_line.d);
      dzs.push_back(dz);
    }
    om_line.dl = *std::min_element(begin(dzs), end(dzs)) - DOM_RADIUS;
    om_line.dh = *std::max_element(begin(dzs), end(dzs)) + DOM_RADIUS;

    om_line.max = positions.size() - 1;
    
    optical_module_lines_[string_number] = om_line;
  }  
}  

void OpticalModuleLines::calculate_sin12(){
  // what is rx and cb?
  const float CV{FPI/180};
  float bv[2][2];
  // DIR1 and DIR2 are magic constants
  bv[0][0] = cos(CV*DIR1); // DIR1 = 9.3
  bv[0][1] = sin(CV*DIR1);
  bv[1][0] = cos(CV*DIR2); // DIR2 = 129.3
  bv[1][1] = sin(CV*DIR2);
  
  sin12_ = 0.;
  for(int i=0; i<2; i++) 
    sin12_ += bv[0][i] * bv[1][i];
  sin12_ = sqrt(1-sin12_*sin12_); //ugh
  // FIXME: scale rx by sin12_ at some point
  //configuration.rx /= sin12_;
}

void OpticalModuleLines::calculate_cb(){

  const float CV{FPI/180};
  float bv[2][2];
  // DIR1 and DIR2 are magic constants
  bv[0][0] = cos(CV*DIR1); // DIR1 = 9.3
  bv[0][1] = sin(CV*DIR1);
  bv[1][0] = cos(CV*DIR2); // DIR2 = 129.3
  bv[1][1] = sin(CV*DIR2);

  // this is a determinant
  float det = (bv[0][0] * bv[1][1]) - (bv[0][1] * bv[1][0]);
  cb_[0][0] = bv[1][1]/det;
  cb_[0][1] = -bv[0][1]/det;
  cb_[1][0] = -bv[1][0]/det;
  cb_[1][1] = bv[0][0]/det;
}


void OpticalModuleLines::configure_cells(const Geometry& geometry,
					 const Configuration& configuration){
  
  std::vector<float> magic_0;
  std::vector<float> magic_1;
  std::cerr<<"optical_module_lines_.size() = "<<optical_module_lines_.size()<<std::endl;
  for(const auto& p: optical_module_lines_){
    const optical_module_line& om_line = p.second;
  
    float magic_number_0 = cb_[0][0]*om_line.x + cb_[1][0]*om_line.y;
    float magic_number_1 = cb_[0][1]*om_line.x + cb_[1][1]*om_line.y;

    magic_0.push_back(magic_number_0);
    magic_1.push_back(magic_number_1);    
  }

  // float -> char  oops
  // cell lower boundary (cl = "cell low")
  // i think these are cell dimensions
  cl_[0] = *std::min_element(begin(magic_0), end(magic_0));
  cl_[1] = *std::min_element(begin(magic_1), end(magic_1));
  
  // cell upper boundary (ch = "cell high")
  float ch0 = *std::max_element(begin(magic_0), end(magic_0));
  float ch1 = *std::max_element(begin(magic_1), end(magic_1));    
  float crst_[2] = { 
    (CX-1)/(ch0 - cl_[0]),
    (CY-1)/(ch1 - cl_[1]),
  };

  // Need to figure out exactly what role each plays:
  //  1) cb_  (hint: it's a 2x2 matrix)
  //  2) DIR1 and DIR2 (i think these are string plane directions)
  //        9.3 and 129.3 (not orthogonal, but neither are our string planes)
  //  The rest should fall into place once these are grokked.
    
  map<unsigned char, int> cells[CX][CY];
  for(const auto& p: optical_module_lines_){
    const unsigned string_number = p.first;
    const optical_module_line& om_line = p.second;

    // n has to be the cell index
    // and we want to know how many strings
    // are in each cell
    float magic_number_0 = cb_[0][0]*om_line.x + cb_[1][0]*om_line.y;
    float magic_number_1 = cb_[0][1]*om_line.x + cb_[1][1]*om_line.y;
      
    //if(n[m]<0 || n[m] >= d.cn[m]){ 
    //  cerr<<"Error in cell initialization"<<endl; 
    //  exit(1); 
    //}

    int n0 = lroundf((magic_number_0 - cl_[0]) * crst_[0]);
    int n1 = lroundf((magic_number_1 - cl_[1]) * crst_[1]);    
    float d1_0 = fabsf(magic_number_0 - (cl_[0] + (n0 - 0.5f)/crst_[0]));
    float d1_1 = fabsf(magic_number_1 - (cl_[1] + (n1 - 0.5f)/crst_[1]));
    float d2_0 = fabsf(magic_number_0 - (cl_[0] + (n0 + 0.5f)/crst_[0]));
    float d2_1 = fabsf(magic_number_1 - (cl_[1] + (n1 + 0.5f)/crst_[1]));

    // keep track of this.  it's a hint into its function.
    //float d = min(d1, d2) * sin12 - s.r;
    //if(d<0){ 
    //  flag=false; 
    //  cerr<<"Warning: string "<<(int)i->first<<" too close to cell boundary"<<endl; 
    //}
    //
    // cells is an array of maps
    // each cell contains a map<key = string_number, value = count>
    cells[n0][n1][string_number]++;
  }

  // initialize is_ and ls_
  unsigned int pos=0;  // odd choice of variable name
  for(int i=0; i < CX; i++){
    for(int j=0; j < CY; j++){
      auto& c = cells[i][j];
      
      is_[i][j] = (c.size() > 0) ? pos : 0x80;

      for(auto p: c){
	//if(c.size() > 0)
	//  std::cerr<<"["<<i<<","<<j<<"] c.size() = "<<c.size()<<std::endl;
	ls_[pos] = (unsigned char)(p.first-1);
	//std::cerr<<"ls["<<pos<<"] = num["<<int(p.first)<<"] = "<<int(ls_[pos])<<std::endl;
	pos++;
      }
      ls_[pos-1] |= 0x80;
    }  
  }
  std::cerr<<"done."<<std::endl;
}  


// largest string radius
// d.rx=0;  
float
OpticalModuleLines::largest_line_radius() const {
  std::vector<float> radii;
  for(auto p: optical_module_lines_){
    radii.push_back(p.second.r);
  }
  return *std::max_element(begin(radii), end(radii));
}

void OpticalModuleLines::to_device(){
  // FIXME: this might be unsafe
  //        map makes no order guarantee,
  //        but we need one.  come back to this.
  for(auto p: optical_module_lines_){
    om_lines_.push_back(p.second);
  }

  unsigned long sizeof_om_lines{om_lines_.size()*sizeof(optical_module_line)};
  std::cerr<<"allocated "<<sizeof_om_lines
	   <<" bytes ("<<int(sizeof_om_lines/1e3)<<" kB) for optical module lines.\n";
  CHECK_ERROR(cudaMalloc((void**) &__device_ptr, sizeof_om_lines));
  CHECK_ERROR(cudaMemcpy(__device_ptr, om_lines_.data(), sizeof_om_lines, cudaMemcpyHostToDevice));

  // TODO: Transfer the following to the device or even break out into a different struct
  //       sin12_
  //       cb_
  //       cl_
  //       crst_
  //       is_
  //       ls_
}


void OpticalModuleLines::pprint() const {
  cerr<<"*** optical_module_lines struct"<<endl;
  cerr<<"  cl[2] = "<<cl_[0]<<" "<<cl_[1]<<endl;
  cerr<<"  crst[2] = "<<crst_[0]<<" "<<crst_[1]<<endl;
  cerr<<"  cb = "<<endl
      <<"     "<<cb_[0][0]<<" "<<cb_[0][1]<<endl
      <<"     "<<cb_[1][0]<<" "<<cb_[1][1]<<endl;
  cerr<<"  is["<<CX<<"]["<<CY<<"] = ";
  for(unsigned i{0}; i<CX; ++i){
    cerr<<endl<<"    ";
    for(unsigned j{0}; j<CY; ++j){
      cerr<<" "<<int(is_[i][j]);
    }
  }
  cerr<<endl;
  cerr<<"  ls["<<NSTR<<"] = ";
  for(unsigned i{0}; i<NSTR; ++i)
    cerr<<" "<<int(ls_[i]);
  cerr<<endl;

  unsigned i{0};
  for(auto p: optical_module_lines_){
    cerr<<"  sc["<<i<<"] ="
	<<" n="<<p.second.n
	<<" max="<<p.second.max
	<<" x="<<p.second.x
	<<" y="<<p.second.y
	<<" r="<<p.second.r
	<<" h="<<p.second.h
	<<" d="<<p.second.d
	<<" dl="<<p.second.dl
	<<" dh="<<p.second.dh
	<<endl;
    i++;
  }
}
