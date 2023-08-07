#include "imu_types.h"
#include "config.h"

namespace demoam {
namespace IMU {

    const float eps = 1e-4;
    const float accelerometer_noise_density = 0.01;
    const float gyroscope_noise_density = 0.000175;
    const float accelerometer_random_walk = 0.000167;
    const float gyroscope_random_walk = 2.91e-006;


    /* 
    * TODO: 
    * sf = sqrt(mImuFreq);
    * IMU::Calib(Tbc,Ng*sf,Na*sf,Ngw/sf,Naw/sf);
    */
    // covariance of measurements
    Eigen::Matrix3d IMUPreIntegration::GyrMeasCov = Matrix3d::Identity() * gyroscope_noise_density * gyroscope_noise_density; // sigma_g * sigma_g / dt, ~6e-6*10
    Eigen::Matrix3d IMUPreIntegration::AccMeasCov = Matrix3d::Identity() * accelerometer_noise_density * accelerometer_noise_density; // sigma_aw * sigma_aw / dt, ~6e-6

    // covariance of bias random walk
    Eigen::Matrix3d IMUPreIntegration::GyrBiasRWCov = Matrix3d::Identity() * gyroscope_random_walk * gyroscope_random_walk;     // sigma_gw * sigma_gw * dt, ~2e-12
    Eigen::Matrix3d IMUPreIntegration::AccBiasRWCov = Matrix3d::Identity() * accelerometer_random_walk * accelerometer_random_walk;     // sigma_aw * sigma_aw * dt, ~4.5e-8

    Eigen::Matrix3d IMUPreIntegration::RightJacobianSO3(const Eigen::Vector3d &v) {
        double x = v.x(), y = v.y(), z = v.z();
        Eigen::Matrix3d I;
        I.setIdentity();
        const double d2 = x * x + y * y + z * z;
        const double d = sqrt(d2);
        Eigen::Matrix3d W = Sophus::SO3d::hat(v);
        if (d < eps)
        {
            return I;
        }
        else
        {
            return I - W * (1.0 - cos(d)) / d2 + W * W * (d - sin(d)) / (d2 * d);
        }
    }
    Eigen::Matrix3d IMUPreIntegration::InverseRightJacobianSO3(const Eigen::Vector3d &v) {
        double x = v.x(), y = v.y(), z = v.z();
        Eigen::Matrix3d I;
        I.setIdentity();
        const double d2 = x * x + y * y + z * z;
        const double d = sqrt(d2);
        Eigen::Matrix3d W = Sophus::SO3d::hat(v);

        if (d < eps)
        {
            return I;
        }
        else
        {
            return I + W / 2 + W * W * (1.0 / d2 - (1.0 + cos(d)) / (2.0 * d * sin(d)));
        }
    }
    Eigen::Matrix3d IMUPreIntegration::NormalizeRotation(const Eigen::Matrix3d &R) {
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        return svd.matrixU() * svd.matrixV().transpose();
    }



} // namespace demoam
} // namespace demoam