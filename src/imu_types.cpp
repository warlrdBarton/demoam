#include "imu_types.h"
#include "config.h"

namespace demoam {

    IMUPreintegration::IMUPreintegration(const Vector3d &original_bg, const Vector3d &original_ba) : bg_(original_bg), ba_(original_ba) {
        const float ng2 = settings::gyroscope_noise_density * settings::gyroscope_noise_density;
        const float na2 = settings::accelerometer_noise_density * settings::accelerometer_noise_density;
        noise_gyro_acce_.diagonal() << ng2, ng2, ng2, na2, na2, na2;
    }

    void IMUPreintegration::Integrate(const IMU &imu, double dt) {
        Vector3d gyr = imu.gyro_ - bg_;  
        Vector3d acc = imu.acce_ - ba_;  

        dp_ = dp_ + dv_ * dt + 0.5f * dR_.matrix() * acc * dt * dt;
        dv_ = dv_ + dR_ * acc * dt;

        Eigen::Matrix<double, 9, 9> A;
        A.setIdentity();
        Eigen::Matrix<double, 9, 6> B;
        B.setZero();

        Matrix3d acc_hat = SO3d::hat(acc);
        double dt2 = dt * dt;

        A.block<3, 3>(3, 0) = -dR_.matrix() * dt * acc_hat;
        A.block<3, 3>(6, 0) = -0.5f * dR_.matrix() * acc_hat * dt2;
        A.block<3, 3>(6, 3) = dt * Matrix3d::Identity();

        B.block<3, 3>(3, 3) = dR_.matrix() * dt;
        B.block<3, 3>(6, 3) = 0.5f * dR_.matrix() * dt2;

        dP_dba_ = dP_dba_ + dV_dba_ * dt - 0.5f * dR_.matrix() * dt2;                      
        dP_dbg_ = dP_dbg_ + dV_dbg_ * dt - 0.5f * dR_.matrix() * dt2 * acc_hat * dR_dbg_;  
        dV_dba_ = dV_dba_ - dR_.matrix() * dt;                                             
        dV_dbg_ = dV_dbg_ - dR_.matrix() * dt * acc_hat * dR_dbg_;                         

        Vector3d omega = gyr * dt;         
        Matrix3d rightJ = SO3d::jr(omega);  
        SO3d deltaR = SO3d::exp(omega);   
        dR_ = dR_ * deltaR;             

        A.block<3, 3>(0, 0) = deltaR.matrix().transpose();
        B.block<3, 3>(0, 0) = rightJ * dt;

        
        cov_ = A * cov_ * A.transpose() + B * noise_gyro_acce_ * B.transpose();

        dR_dbg_ = deltaR.matrix().transpose() * dR_dbg_ - rightJ * dt; 

        dt_ += dt;
    }

    SO3d IMUPreintegration::GetDeltaRotation(const Vector3d &bg) { return dR_ * SO3d::exp(dR_dbg_ * (bg - bg_)); }

    Vector3d IMUPreintegration::GetDeltaVelocity(const Vector3d &bg, const Vector3d &ba) {
        return dv_ + dV_dbg_ * (bg - bg_) + dV_dba_ * (ba - ba_);
    }

    Vector3d IMUPreintegration::GetDeltaPosition(const Vector3d &bg, const Vector3d &ba) {
        return dp_ + dP_dbg_ * (bg - bg_) + dP_dba_ * (ba - ba_);
    }
/*
    NavStated IMUPreintegration::Predict(const sad::NavStated &start, const Vector3d &grav) const {
        SO3d Rj = start.R_ * dR_;
        Vector3d vj = start.R_ * dv_ + start.v_ + grav * dt_;
        Vector3d pj = start.R_ * dp_ + start.p_ + start.v_ * dt_ + 0.5f * grav * dt_ * dt_;

        auto state = NavStated(start.timestamp_ + dt_, Rj, pj, vj);
        state.bg_ = bg_;
        state.ba_ = ba_;
        return state;
    }
*/

    /* 
    * TODO: 
    * sf = sqrt(mImuFreq);
    * IMU::Calib(Tbc,Ng*sf,Na*sf,Ngw/sf,Naw/sf);
    * *10, *100
    */


   /*
    // covariance of measurements
    Eigen::Matrix3d IMUPreintegration::GyrMeasCov = Matrix3d::Identity() * gyroscope_noise_density * gyroscope_noise_density; // sigma_g * sigma_g / dt, ~6e-6*10
    Eigen::Matrix3d IMUPreintegration::AccMeasCov = Matrix3d::Identity() * accelerometer_noise_density * accelerometer_noise_density; // sigma_aw * sigma_aw / dt, ~6e-6

    // covariance of bias random walk
    Eigen::Matrix3d IMUPreintegration::GyrBiasRWCov = Matrix3d::Identity() * gyroscope_random_walk * gyroscope_random_walk;     // sigma_gw * sigma_gw * dt, ~2e-12
    Eigen::Matrix3d IMUPreintegration::AccBiasRWCov = Matrix3d::Identity() * accelerometer_random_walk * accelerometer_random_walk;     // sigma_aw * sigma_aw * dt, ~4.5e-8

    Eigen::Matrix3d IMUPreintegration::RightJacobianSO3(const Eigen::Vector3d &v) {
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
    Eigen::Matrix3d IMUPreintegration::InverseRightJacobianSO3(const Eigen::Vector3d &v) {
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
    Eigen::Matrix3d IMUPreintegration::NormalizeRotation(const Eigen::Matrix3d &R) {
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        return svd.matrixU() * svd.matrixV().transpose();
    }

*/

} // namespace demoam