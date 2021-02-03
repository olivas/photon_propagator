#include <map>
#include <deque>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

#include <photon_propagator/configuration.hpp>
#include <photon_propagator/cuda/check_error.cuh>

using std::string;
using std::cerr;
using std::endl;
using std::ifstream;
using std::vector;

const float DOM_RADIUS{0.16510}; // DOM radius [m]

using boost::filesystem::path;
using boost::filesystem::exists;

Configuration::Configuration(const path& ice_model_path)
{
  configuration_filename_ = ice_model_path;
  angular_sensitivity_filename_ = ice_model_path;
  configuration_filename_ /= "cfg.txt";
  angular_sensitivity_filename_ /= "as.dat";
  
  assert(exists(configuration_filename_));
  assert(exists(angular_sensitivity_filename_));
  
  configure(configuration_filename_);
  configure_angular_sensitivity(angular_sensitivity_filename_);
}

void Configuration::to_device(){
  // where is line configured
  // we're down to 3k for the configuration size.
  // used to be 7k (7824 bytes).
  // we're missing something.
  // we're missing the anisotropy and tilt parameters
  unsigned long sizeof_configuration{sizeof(configuration)};
  std::cerr<<"allocated "<<sizeof_configuration
	   <<" bytes ("<<int(sizeof_configuration/1e3)<<" kB) for configuration.\n";
  CHECK_ERROR(cudaMalloc((void**) &__device_ptr, sizeof_configuration));
  CHECK_ERROR(cudaMemcpy(__device_ptr, &host_configuration_, sizeof_configuration, cudaMemcpyHostToDevice));  
}

void Configuration::configure(const path& configuration_filename){
  cerr<<"Configuration in "<<configuration_filename.c_str()<<endl;
  ifstream infile(configuration_filename.c_str());
  if(!infile.fail()){
    string in;
    float aux;
    vector<float> v;
    while(getline(infile, in)) 
      if(sscanf(in.c_str(), "%f", &aux)==1) 
	v.push_back(aux);
    
    if(v.size()>=4){
      xR_ = lroundf(v[0]); 
      host_configuration_.eff = v[1];
      host_configuration_.sf = v[2];
      host_configuration_.g = v[3];
      
      host_configuration_.zR = 1.f/xR_;
      //photon_bunch_size *= xR*xR; FIXME
      host_configuration_.R = DOM_RADIUS*xR_; 
      host_configuration_.R2 = host_configuration_.R*host_configuration_.R;
      host_configuration_.g2 = host_configuration_.g*host_configuration_.g; 
      host_configuration_.gr = (1-host_configuration_.g)/(1+host_configuration_.g);
      cerr<<"Configured: xR="<<xR_
	  <<" eff="<<host_configuration_.eff
	  <<" sf="<<host_configuration_.sf
	  <<" g="<<host_configuration_.g<<endl;      
    }else{
      cerr<<"File cfg.txt did not contain valid data"<<endl;
      exit(1);
    }    
    infile.close();
  }else{  
    cerr<<"Could not open file cfg.txt"<<endl;
    exit(1);
  }
}

void
Configuration::configure_angular_sensitivity(const path& angular_sensitivity_filename){
  host_configuration_.mas=1;
  
  host_configuration_.s[0]=1;
  for(int i=1; i<ANUM; i++) 
    host_configuration_.s[i]=0;
  
  ifstream infile(angular_sensitivity_filename.c_str());
  infile >> host_configuration_.mas;
  for(int i=0; i<ANUM; i++){
    infile >> host_configuration_.s[i];
  }
  infile.close();
  
  // d.mas is maximum angular sensitivity
  // FIXME: scale the eff somewhere else.
  host_configuration_.eff *= host_configuration_.mas;
}

void Configuration::pprint(){
  cerr<<"*** configuration struct"<<endl;
  cerr<<"  sf = "<<host_configuration_.sf<<endl
      <<"  g = "<<host_configuration_.g<<endl
      <<"  g2 = "<<host_configuration_.g2<<endl
      <<"  gr = "<<host_configuration_.gr<<endl
      <<"  R = "<<host_configuration_.R<<endl
      <<"  R2 = "<<host_configuration_.R2<<endl
      <<"  zR = "<<host_configuration_.zR<<endl
      <<"  eff = "<<host_configuration_.eff<<endl
      <<"  mas = "<<host_configuration_.mas<<endl;  
  cerr<<"  s["<<ANUM<<"] = "<<endl;
  cerr<<"   ";
  for(unsigned i{0}; i<ANUM; ++i)
    cerr<<" "<<host_configuration_.s[i];
  cerr<<endl;
      
}
