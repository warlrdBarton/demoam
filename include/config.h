#ifndef DEMOAM__CONFIG_H
#define DEMOAM__CONFIG_H

#include "common_include.h"
#include "geometric_tools.h"

namespace demoam {
namespace settings {

    const float GRAVITY_VALUE = 9.81;
    const float eps = 1e-4;

    const float accelerometer_noise_density = 0.01;
    const float gyroscope_noise_density = 0.000175;
    const float accelerometer_random_walk = 0.000167;
    const float gyroscope_random_walk = 2.91e-006;

    const double gyrBiasRw2 = gyroscope_random_walk * gyroscope_random_walk;   
    const double accBiasRw2 = accelerometer_random_walk * accelerometer_random_walk;  
    const double gyrMeasError2 = gyroscope_noise_density * gyroscope_noise_density;   
    const double accMeasError2 = accelerometer_noise_density * accelerometer_noise_density;  

/*     const double gyrBiasRw2 = gyroscope_random_walk * gyroscope_random_walk * 100;   
    const double accBiasRw2 = accelerometer_random_walk * accelerometer_random_walk * 100;  
    const double gyrMeasError2 = gyroscope_noise_density * gyroscope_noise_density / 0.005;   
    const double accMeasError2 = accelerometer_noise_density * accelerometer_noise_density / 0.005;   */

    const float keyframeTimeGapTracking = 3.0f;

    const Sophus::SE3d Tcb = []{
        Sophus::SE3d Tvi, Tiv; // per Kitti calibration
        {
            Eigen::Matrix3d R;
            Eigen::Vector3d t;
            R << 9.999976e-01, 7.553071e-04, -2.035826e-03, 
                -7.854027e-04, 9.998898e-01, -1.482298e-02,
                2.024406e-03, 1.482454e-02, 9.998881e-01;
            R = NormalizeRotation(R);
            t << -8.086759e-01, 3.195559e-01, -7.997231e-01;
            Tvi = Sophus::SE3d(R, t);
            Tiv = Tvi.inverse();
        }
        Sophus::SE3d Tcv, Tvc;
        {
            Eigen::Matrix3d R;
            Eigen::Vector3d t;
            R << 7.027555e-03, -9.999753e-01, 2.599616e-05,
                -2.254837e-03, -4.184312e-05, -9.999975e-01,
                9.999728e-01, 7.027479e-03, -2.255075e-03;
            R = NormalizeRotation(R);
            t << -7.137748e-03, -7.482656e-02, -3.336324e-01;
            Tcv = Sophus::SE3d(R, t);
            Tvc = Tcv.inverse();
        }
        Sophus::SE3d Tci, Tci_inv;
        Tci = Tcv * Tvi;
        Tci_inv = Tci.inverse();
        return Tci;
    }();

    const Sophus::SE3d Tbc = Tcb.inverse();

    const Eigen::Vector3d biasAccePrior(-0.025, 0.136, 0.075);

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