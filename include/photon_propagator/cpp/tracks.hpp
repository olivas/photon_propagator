#pragma once

#include <vector>
#include <memory>

#include <photon_propagator/cpp/device.hpp>
#include <photon_propagator/cpp/particle.hpp>

struct track;

class Tracks {
public:

    /*
    struct Position {
        float x;
        float y;
        float z;
        float time;
    };

    struct Direction {
        float x;
        float y;
        float z;
    };

    struct Parameters{
        float l;
        float f;
    };
*/
    Tracks(const size_t count, const std::shared_ptr <Device> &device);

    ~Tracks();

    void add(const particle &p);

    size_t nbytes() const;

    size_t n_photons() const;

    void to_device();

    track *__device_ptr;

    void pprint() const {};

private:

    size_t count_;
    std::vector <track> host_tracks_;
    const std::shared_ptr <Device> &device_;

};

