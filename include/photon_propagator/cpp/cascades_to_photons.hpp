
class Random;
class IceModel;
class Cascades;
class Device;
class Photons;

namespace cascades{

  /**
   * Launches a single kernel and generates n_concurrent_thread
   * photons.  Used mostly for testing.
   */ 
  void GeneratePhotons(const Random& rng,
		       const IceModel& ice_model,
		       const Cascades& cascades,
		       const std::shared_ptr<Device>& device,
		       const unsigned number_of_blocks,
		       const unsigned threads_per_block,
		       Photons& output);
}
