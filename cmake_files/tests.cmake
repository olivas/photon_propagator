
enable_testing()

set(ICE_MODEL_PATH ../resources/ice/mie)
set(GEOMETRY_FILE ../resources/detector/icecube.txt)

add_executable(test_random tests/test_random.cu)
set_target_properties(test_random
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)
set_target_properties(test_random
        PROPERTIES POSITION_INDEPENDENT_CODE ON
)
set_property(TARGET test_random PROPERTY CUDA_ARCHITECTURES OFF)

target_link_libraries(test_random photon_propagator_cuda ${Boost_LIBRARIES})
add_test(test_random bin/test_random)

#add_executable(test_configuration tests/test_configuration.cpp)
#target_link_libraries(test_configuration photon_propagator_cuda ${Boost_LIBRARIES})
#add_test(test_configuration bin/test_configuration ${ICE_MODEL_PATH})
#
#add_executable(test_ice_model tests/test_ice_model.cpp)
#target_link_libraries(test_ice_model photon_propagator_cuda ${Boost_LIBRARIES})
#add_test(test_ice_model bin/test_ice_model ${ICE_MODEL_PATH})
#
#add_executable(test_geometry tests/test_geometry.cpp)
#target_link_libraries(test_geometry photon_propagator_cuda ${Boost_LIBRARIES})
#add_test(test_geometry bin/test_geometry ${GEOMETRY_FILE})
#
#add_executable(test_optical_module_lines tests/test_optical_module_lines.cpp)
#target_link_libraries(test_optical_module_lines photon_propagator_cuda ${Boost_LIBRARIES})
#add_test(test_optical_module_lines bin/test_optical_module_lines ${GEOMETRY_FILE} ${ICE_MODEL_PATH})
#
#add_executable(test_hits tests/test_hits.cpp)
#target_link_libraries(test_hits photon_propagator_cuda ${Boost_LIBRARIES})
#add_test(test_hits bin/test_hits)
#
#add_executable(test_photons tests/test_photons.cpp)
#target_link_libraries(test_photons photon_propagator_cuda ${Boost_LIBRARIES})
#add_test(test_photons bin/test_photons)
#
#add_executable(test_rotate tests/test_rotate.cpp)
#target_link_libraries(test_rotate photon_propagator_cuda ${Boost_LIBRARIES})
#add_test(test_rotate bin/test_rotate)
#
#add_executable(test_tracks tests/test_tracks.cpp)
#target_link_libraries(test_tracks photon_propagator_cuda ${Boost_LIBRARIES})
#add_test(test_tracks bin/test_tracks)
#
#add_executable(test_cascades tests/test_cascades.cpp)
#target_link_libraries(test_cascades photon_propagator_cuda ${Boost_LIBRARIES})
#add_test(test_cascades bin/test_cascades)
#
#add_executable(test_cascades_to_photons tests/test_cascades_to_photons.cpp)
#target_link_libraries(test_cascades_to_photons photon_propagator_cuda ${Boost_LIBRARIES})
#add_test(test_cascades_to_photons bin/test_cascades_to_photons)
#
##add_executable(test_propagation tests/test_propagation.cpp)
##target_link_libraries(test_propagation photon_propagator_cuda photon_propagator ${Boost_LIBRARIES})
##add_test(test_propagation bin/test_propagation ${GEOMETRY_FILE} ${ICE_MODEL_PATH})

