#pragma once

#include <string>
#include <array>
#include <cmath>

template <unsigned N>
class histogram{
public:
  histogram(double xmin, double xmax, std::string name):
    xmin_(xmin),
    xmax_(xmax),
    name_(name)
  {
    for(auto& bv: bin_values_){
      bv = 0;
    }
    inv_bw_ = N/(xmax_-xmin_);    
  }

  void fill(double value){
    if(value >= xmin_ and value < xmax_){
      bin_values_[inv_bw_ * (value - xmin_)]++;
    }else{
      if(std::isnan(value)){
	nan_count_++;
      }else{
	if(value >= xmax_){	  
	  overflow_++;
	}else if(value < xmin_){
	  underflow_++;
	}
      }
    }
  }
  
private:
  double xmin_;
  double xmax_;
  std::string name_;
  std::array<unsigned, N> bin_values_;
  
  double overflow_;
  double underflow_;
  double nan_count_;

  double inv_bw_;

  // write in json
  friend std::ostream& operator<<(std::ostream &os, const histogram& h){
    os << "{\"histogram\" : { \"xmin\" : "<< h.xmin_ << ","
       << " \"xmax\" : "<< h.xmax_ << ","
       << " \"name\" : "<< h.name_ << ","
       << " \"bin_values\" : [";
    for(auto bv: h.bin_values_)
      os << bv << ",";
    os << "] ,";
    os << " \"overflow\" : "<< h.overflow_ << ","
       << " \"underflow\" : "<< h.underflow_ << ","
       << " \"nan_count\" : "<< h.nan_count_ << ","
       << " \"inv_bw\" : "<< h.inv_bw_ << "}}"<<std::endl;  
  }
  
};

