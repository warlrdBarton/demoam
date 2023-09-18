#ifndef DEMOAM__CONFIG_H
#define DEMOAM__CONFIG_H

#include "common_include.h"

namespace demoam {
namespace settings {

    const float GRAVITY_VALUE = 9.81;
    const float eps = 1e-4;
    const float accelerometer_noise_density = 0.01;
    const float gyroscope_noise_density = 0.000175;
    const float accelerometer_random_walk = 0.000167;
    const float gyroscope_random_walk = 2.91e-006;

    const int NUM_KFs_FOR_IMU_INIT = 5;

} // namespace settings

class Config {
 public:
    ~Config();
    static bool SetParameterFile(const std::string& filename);
    template <typename T> 
    static T Get(const std::string &key) {
        return T(Config::config_ -> file_[key]);
    }

 private:
    static std::shared_ptr<Config> config_;
    cv::FileStorage file_;
    Config() {} // private constructor makes a singleton
};

} // namespace demoam

#endif // DEMOAM__CONFIG_H