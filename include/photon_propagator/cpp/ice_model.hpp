#include <string>
#include <vector>

#include <boost/filesystem.hpp>
#include <photon_propagator/device_structs/ice_color.h>

class IceModel{
public:
  IceModel(const boost::filesystem::path& ice_model_path, float scattering_angle);

  void to_device();  
  ice_color* __device_ptr; // array of ice_color objects

  size_t n_layers(){ return n_layers_; }
  float layers_spacing(){ return layer_spacing_; }
  float hmin(){ return hmin_; }
  float c_vacuum(){ return c_; }

  void pprint() const;

  size_t size_of() const { return color_slices_.size()*sizeof(ice_color); }
  
private:

  boost::filesystem::path ice_model_parameters_filename_;
  boost::filesystem::path ice_model_filename_;
  boost::filesystem::path wavelength_data_filename_;
  size_t n_layers_;
  float layer_spacing_;
  float hmin_;
  float c_;
  float g_; // scattering angle
  std::vector<ice_color> color_slices_;
  
  void configure(const boost::filesystem::path& ice_model_parameters_filename,
		 const boost::filesystem::path& ice_model_filename,
		 const boost::filesystem::path& wavelength_data_filename);		   

};
