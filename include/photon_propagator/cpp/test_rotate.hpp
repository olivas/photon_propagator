
#include <array>
#include <vector>

#include <photon_propagator/random.hpp>

namespace test{

  void test_rotations(std::vector<std::array<float, 3>>& directions,		    
		      const unsigned long long number_of_blocks,
		      const unsigned long long threads_per_block,
		      const std::shared_ptr<Random>& rng,
		      const float cs,
		      const float si);
}
