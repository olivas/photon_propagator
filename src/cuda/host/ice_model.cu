
#include <fstream>
#include <iostream>
#include <photon_propagator/cpp/ice_model.hpp>

#include <photon_propagator/cuda/check_error.cuh>

const float C{0.299792458};             // speed of light in a vacuum
const float ZOFF{1948.07};              // offset in Z obvs
const unsigned N_WAVELENGTH_SLICES{32}; // number of wavelength slices

using std::string;
using std::ifstream;
using std::vector;
using std::cerr;
using std::endl;
using boost::filesystem::path;
using boost::filesystem::exists;

IceModel::IceModel(const path& ice_model_path, float scattering_angle):
  g_(scattering_angle),
  c_(C)
{
  ice_model_parameters_filename_ = ice_model_path;
  ice_model_filename_ = ice_model_path;
  wavelength_data_filename_ = ice_model_path;
  
  ice_model_parameters_filename_ /= "icemodel.par";
  ice_model_filename_ /= "icemodel.dat";
  wavelength_data_filename_ /= "wv.dat";
  
  assert(exists(ice_model_parameters_filename_));
  assert(exists(ice_model_filename_));
  assert(exists(wavelength_data_filename_));
  
  configure(ice_model_parameters_filename_,
	    ice_model_filename_,
	    wavelength_data_filename_);
}

void IceModel::to_device(){
  unsigned long sizeof_ice_model{color_slices_.size()*sizeof(ice_color)};
  std::cerr<<"allocated "<<sizeof_ice_model
	   <<" bytes ("<<int(sizeof_ice_model/1e3)<<" kB) for the ice model.\n";
  CHECK_ERROR(cudaMalloc((void**) &__device_ptr, sizeof_ice_model));
  CHECK_ERROR(cudaMemcpy(__device_ptr, color_slices_.data(), sizeof_ice_model, cudaMemcpyHostToDevice));  
}

void IceModel::configure(const path& ice_model_parameters_filename,
			 const path& ice_model_filename,
			 const path& wavelength_data_filename){
  
  ifstream ice_model_par_file(ice_model_parameters_filename_.c_str());
  float wv0=400;
  float A, B, D, E, a, k;
  float Ae, Be, De, Ee, ae, ke;
  ice_model_par_file >> a >> ae;
  ice_model_par_file >> k >> ke;
  ice_model_par_file >> A >> Ae;
  ice_model_par_file >> B >> Be;
  ice_model_par_file >> D >> De;
  ice_model_par_file >> E >> Ee;
  ice_model_par_file.close();
  D=pow(wv0, k);
  
  ifstream ice_model_file(ice_model_filename.c_str());
  float dpa, bea, baa, tda;
  vector<float> dp, be, ba, td;  
  while(ice_model_file >> dpa >> bea >> baa >> tda){
    dp.push_back(dpa);
    be.push_back(bea);
    ba.push_back(baa);
    td.push_back(tda);
  }
  ice_model_file.close();
  cerr<<"Loaded "<<dp.size()<<" ice layers"<<endl;
  
  layer_spacing_ = dp[1]-dp[0];
  n_layers_ = dp.size();
  hmin_ = ZOFF - dp[n_layers_-1];
  
  ifstream wv_dat_file(wavelength_data_filename.c_str());
  vector<float> wx; // efficiency
  vector<float> wy; // wavelength
  if(!wv_dat_file.fail()){
    float xa;
    float ya;
    while(wv_dat_file>>xa>>ya){
      wx.push_back(xa);
      wy.push_back(ya);
    }
    wv_dat_file.close();
    cerr<<"Loaded "<<wx.size()<<" wavelength points"<<endl;
  }else{
    cerr<<"Could not open file wv.dat"<<endl;
    exit(1);
  }

  // it's important to insert in the proper order
  // since this is pushed onto the device.
  // std::vector elements live in contiguous memory
  // so we're safe.
  color_slices_.clear();
  for(int n=0; n<N_WAVELENGTH_SLICES; n++){
    float p=(n+0.5f)/N_WAVELENGTH_SLICES;
    int m=0;
    while(wx[++m] < p);
    
    float wva=(wy[m-1]*(wx[m]-p)+wy[m]*(p-wx[m-1]))/(wx[m]-wx[m-1]);        
    float l_a=pow(wva/wv0, -a);
    float l_k=pow(wva, -k);
    float ABl=A*exp(-B/wva);
    
    ice_color slice;    
    for(int i=0; i<n_layers_; i++){
      int j=n_layers_-1-i;
      float sca=(be[j]*l_a)/(1-g_);
      float abs=(D*ba[j]+E)*l_k+ABl*(1+0.01*td[j]);
      slice.ice_properties[i].sca = sca;
      slice.ice_properties[i].abs = abs;
    }
    
    float wv = wva*1.e-3;
    float wv2 = wv*wv;
    float wv3 = wv*wv2;
    float wv4 = wv*wv3;
    float np = 1.55749-1.57988*wv+3.99993*wv2-4.68271*wv3+2.09354*wv4;
    float ng = np*(1+0.227106-0.954648*wv+1.42568*wv2-0.711832*wv3);
        
    slice.wvl = wva;
    slice.ocm = ng/C;
    slice.coschr = 1/np;
    slice.sinchr = sqrt(1 - slice.coschr*slice.coschr);
    color_slices_.push_back(slice);
  }
  
};

void IceModel::pprint() const {
  cerr<<"*** ice model struct"<<endl;
  cerr<<"  size = "<<n_layers_<<endl
      <<"  dh = "<<layer_spacing_<<endl
      <<"  hdh = "<<layer_spacing_/2.<<endl
      <<"  rdh = "<<1./layer_spacing_<<endl
      <<"  hmin = "<<hmin_<<endl
      <<"  ocv = "<<1/c_<<endl;
}
