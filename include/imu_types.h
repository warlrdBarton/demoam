#ifndef DEMOAM__IMU_TYPES_H
#define DEMOAM__IMU_TYPES_H

#include "common_include.h"

namespace demoam
{


// gyro, accelerometer and timestamp
struct IMU {
    IMU() = default;
    IMU(double t, const Vector3d& gyro, const Vector3d& acce) : timestamp_(t), gyro_(gyro), acce_(acce) {}
    IMU(double t, 
        const double &ang_vel_x, const double &ang_vel_y, const double &ang_vel_z,
        const double &acce_x, const double &acce_y, const double &acce_z) 
        : timestamp_(t), gyro_(ang_vel_x, ang_vel_y, ang_vel_z), acce_(acce_x, acce_y, acce_z) {}
        
    double timestamp_ = 0.0;
    Vector3d gyro_ = Vector3d::Zero();
    Vector3d acce_ = Vector3d::Zero();
};

using IMUPtr = std::shared_ptr<IMU>;

/*
* IMU PreIntegrator
* Call Integrate to insert new IMU reading, call Getters for preintegrated measurements 
* Jacobian are also computed in this class, which can be used to edges of g2o
*/
class IMUPreintegration {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    IMUPreintegration(const Vector3d &original_bg = Vector3d::Zero(), const Vector3d &original_ba = Vector3d::Zero());

    void Integrate(const IMU &imu, double dt);


   // NavStated Predict(const NavStated &start, const Vector3d &grav = Vector3d(0, 0, -9.81)) const;

    SO3d GetDeltaRotation(const Vector3d &bg);
    Vector3d GetDeltaVelocity(const Vector3d &bg, const Vector3d &ba);
    Vector3d GetDeltaPosition(const Vector3d &bg, const Vector3d &ba);

   public:
    double dt_ = 0;                          
    Matrix9d cov_ = Matrix9d::Zero();              
    Matrix6d noise_gyro_acce_ = Matrix6d::Zero();  

    // Bias used when computing preintegration
    // the updated val stores in each frame;
    Vector3d bg_ = Vector3d::Zero();
    Vector3d ba_ = Vector3d::Zero();

    SO3d dR_;
    Vector3d dv_ = Vector3d::Zero();
    Vector3d dp_ = Vector3d::Zero();

    Matrix3d dR_dbg_ = Matrix3d::Zero();
    Matrix3d dV_dbg_ = Matrix3d::Zero();
    Matrix3d dV_dba_ = Matrix3d::Zero();
    Matrix3d dP_dbg_ = Matrix3d::Zero();
    Matrix3d dP_dba_ = Matrix3d::Zero();
};

} // namespace demoam

#endif // DEMOAM__IMU_TYPES_H
