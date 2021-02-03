#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <memory>
#include <cassert>
#include <cstdlib>

#include <boost/filesystem.hpp>

#include <photon_propagator/detector.hpp>
#include <photon_propagator/histogram/histogram.hpp>
#include <photon_propagator/photon_propagator.hpp>

using boost::filesystem::path;
using boost::filesystem::exists;

void load_detector(std::ifstream& ifs, detector& d){
  std::string line;
  // the format is str, om, x, y, z, hv, rde
  while(std::getline(ifs, line)){
    std::stringstream sstr(line);
    optical_module om;
    sstr >> om.str;
    sstr >> om.dom;
    sstr >> om.x;
    sstr >> om.y;
    sstr >> om.z;
    sstr >> om.hv;
    sstr >> om.rde;
    d.push_back(om);
  }
}


class event_file_reader{
public:
  event_file_reader(std::ifstream& ifs): ifs_(ifs){}

  std::unique_ptr<event> get_event(){
    std::string line;
    if(std::getline(ifs_, line)){
      std::stringstream sstr(line);

      int iptype;      
      particle p;
      sstr >> p.major_id;
      sstr >> p.minor_id;
      sstr >> iptype; 
      sstr >> p.time;
      sstr >> p.energy;
      sstr >> p.length;
      sstr >> p.direction[0];
      sstr >> p.direction[1];
      sstr >> p.direction[2];
      sstr >> p.position[0];
      sstr >> p.position[1];
      sstr >> p.position[2];
      
      p.ptype = static_cast<particle_type>(iptype);

      event e;
      e.push_back(p);
      return std::unique_ptr<event>(new event(e));
    };
    return nullptr;
  }
  
private:

  std::ifstream& ifs_;
};

int main(int argc, char* argv[]){

  if(argc != 2){
    std::cerr<<"Usage: generator <filename>"<<std::endl;
    std::exit(-1);
  }

  path input_file_path(argv[1]);
  if(!exists(input_file_path)){
    std::cerr<<"ERROR: Input path "<<input_file_path<<" does not exist."<<std::endl;
    std::exit(-1);
  }

  const std::string PPFTP_PATH(std::getenv("PPFTP_PATH"));
  assert(!PPFTP_PATH.empty());
  
  std::string detector_filename(PPFTP_PATH + "/resources/detector/icecube.txt");
  assert(exists(path(detector_filename)));
  
  std::ifstream detector_file(detector_filename);  
  detector icecube; // using detector = std::vector<optical_module>
  load_detector(detector_file, icecube);
			    
  photon_propagator propagator(icecube);

  // now we read particles from a text file
  // and generate histograms
  std::ifstream ifs(input_file_path.c_str());
  event_file_reader file_reader(ifs);

  histogram<64> pe_dom_occup(1, 65, "PEDOMOccup");
  histogram<1000> n_mc_pe_chan(0, 1000, "NMCPEChan");
  histogram<1000> mc_pe_time(-10000, 10000, "MCPETime");
  histogram<100> mc_pes(0, 1000, "MCPEs");

  particle_type ptype = particle_type::unset;

  unsigned event_counter = 0;
  std::unique_ptr<event> ev;;
  while(ev = file_reader.get_event()){
    event_counter++;

    if(ptype == particle_type::unset)
      ptype = ev->front().ptype;
    
    photon_arrival_times pe_times;
    propagator.process_event(*ev, pe_times);

    n_mc_pe_chan.fill(pe_times.size());
    for(auto& p: pe_times){
      pe_dom_occup.fill(p.first.second);

      mc_pes.fill(p.second.size());
      for(double t: p.second){
	mc_pe_time.fill(t);
      }
    }
  }

  // write histograms to a text file in JSON format
  std::string filename = std::string("test_") + stringize(ptype) + std::string(".json");
  std::ofstream ofs("test.json"); // file name is test_<type>.json
  ofs << pe_dom_occup
      << n_mc_pe_chan
      << mc_pe_time
      << mc_pes;

  // the python validation script will read this text file, 
  // fill histogram classes, compare them to benchmarks,
  // and generate plots with both histograms with ratios and colors
  // red filled histograms means FAIL.  green means pass. 
}

  
