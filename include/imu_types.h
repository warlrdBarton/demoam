#ifndef DEMOAM__IMU_TYPES_H
#define DEMOAM__IMU_TYPES_H

#include "common_include.h"

namespace demoam
{

namespace IMU
{

const float GRAVITY_VALUE=9.81;

// gyro, accelerometer and timestamp
class Point
{
public:
    Point(const float &acc_x, const float &acc_y, const float &acc_z,
             const float &ang_vel_x, const float &ang_vel_y, const float &ang_vel_z,
             const double &timestamp): a(acc_x,acc_y,acc_z), w(ang_vel_x,ang_vel_y,ang_vel_z), t(timestamp){}
    Point(const cv::Point3f Acc, const cv::Point3f Gyro, const double &timestamp):
        a(Acc.x,Acc.y,Acc.z), w(Gyro.x,Gyro.y,Gyro.z), t(timestamp){}
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3f a;
    Eigen::Vector3f w;
    double t;
};


class IMUPreIntegration {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    IMUPreIntegration() {
        reset();
    }
    // reset to initial state
    inline void reset() {
        // delta measurements, position/velocity/rotation(matrix)
        _delta_P.setZero();    // P_k+1 = P_k + V_k*dt + R_k*a_k*dt*dt/2
        _delta_V.setZero();    // V_k+1 = V_k + R_k*a_k*dt
        _delta_R.setIdentity();    // R_k+1 = R_k*exp(w_k*dt).     note: Rwc, Rwc'=Rwc*[w_body]x
        // jacobian of delta measurements w.r.t bias of gyro/acc
        _J_P_Biasg.setZero();     // position / gyro
        _J_P_Biasa.setZero();     // position / acc
        _J_V_Biasg.setZero();     // velocity / gyro
        _J_V_Biasa.setZero();     // velocity / acc
        _J_R_Biasg.setZero();   // rotation / gyro
        // noise covariance propagation of delta measurements
        _cov_P_V_Phi.setZero();
        _delta_time = 0;
    }
    // incrementally update 1)delta measurements, 2)jacobians, 3)covariance matrix
    void update(const Eigen::Vector3d &omega, const Eigen::Vector3d &acc, const double &dt) {
        float dt2 = dt * dt;
        Eigen::Matrix3d dR = Sophus::SO3d::exp(omega * dt).matrix();
        //Eigen::Matrix3d Jr = Sophus::SO3d::JacobianR(omega * dt).matrix();
        Eigen::Matrix3d Jr = RightJacobianSO3(omega * dt);
        Eigen::Matrix3d I3x3 = Eigen::Matrix3d::Identity();
        Eigen::Matrix<double, 9, 9> A = Eigen::Matrix<double, 9, 9>::Identity();
        Eigen::Matrix3d acc_h = Sophus::SO3d::hat(acc).matrix();
        // noise propagation, see Forster's paper APPENDIX A.
        A.block<3, 3>(6, 6) = dR.transpose();
        A.block<3, 3>(3, 6) = -_delta_R * acc_h * dt;
        A.block<3, 3>(0, 6) = -0.5 * _delta_R * acc_h * dt2;
        A.block<3, 3>(0, 3) = I3x3 * dt;
        Eigen::Matrix<double, 9, 3> Bg;
        Bg.setZero();
        Bg.block<3, 3>(6, 0) = Jr * dt;
        Eigen::Matrix<double, 9, 3> Ca;
        Ca.setZero();
        Ca.block<3, 3>(3, 0) = -_delta_R * dt;
        Ca.block<3, 3>(0, 0) = 0.5 * _delta_R * dt2;
        _cov_P_V_Phi = A * _cov_P_V_Phi * A.transpose()
                       + Bg * GyrMeasCov * Bg.transpose()
                       + Ca * AccMeasCov * Ca.transpose();
        // calculate the jacobians
        // NOTE see Forster's paper appendix B. Bias correction via first order updates
        _J_P_Biasa += _J_V_Biasa * dt - 0.5 * _delta_R * dt2;
        _J_P_Biasg += _J_V_Biasg * dt - 0.5 * _delta_R * acc_h * _J_R_Biasg * dt2;
        _J_V_Biasa += -_delta_R * dt;
        _J_V_Biasg += -_delta_R * acc_h * _J_R_Biasg * dt;
        _J_R_Biasg = dR.transpose() * _J_R_Biasg - Jr * dt;
        // delta measurements, position/velocity/rotation(matrix)
        // update P first, then V, then R. because P's update need V&R's previous state
        _delta_P += _delta_V * dt + 0.5 * _delta_R * acc * dt2;    // P_k+1 = P_k + V_k*dt + R_k*a_k*dt*dt/2
        _delta_V += _delta_R * acc * dt;
        _delta_R = normalizeRotationM(
                _delta_R * dR);  // normalize rotation, in case of numerical error accumulation
        // delta time
        _delta_time += dt;
    }
    // normalize rotation
    inline Eigen::Matrix3d normalizeRotationM(const Eigen::Matrix3d &R) {
        Quaterniond qr(R);
        return normalizeRotationQ(qr).toRotationMatrix();
    }
    inline Eigen::Quaterniond normalizeRotationQ(const Eigen::Quaterniond &r) {
        Eigen::Quaterniond _r(r);
        if (_r.w() < 0) {
            _r.coeffs() *= -1;
        }
        return _r.normalized();
    }
    // accessors
    // delta measurements, position/velocity/rotation(matrix)
    inline Eigen::Vector3d getDeltaP() const    // P_k+1 = P_k + V_k*dt + R_k*a_k*dt*dt/2
    {
        return _delta_P;
    }
    inline Eigen::Vector3d getDeltaV() const    // V_k+1 = V_k + R_k*a_k*dt
    {
        return _delta_V;
    }
    inline Eigen::Matrix3d getDeltaR() const   // R_k+1 = R_k*exp(w_k*dt).     NOTE: Rwc, Rwc'=Rwc*[w_body]x
    {
        return _delta_R;
    }
    // jacobian of delta measurements w.r.t bias of gyro/acc
    inline Eigen::Matrix3d getJPBiasg() const     // position / gyro
    {
        return _J_P_Biasg;
    }
    inline Eigen::Matrix3d getJPBiasa() const     // position / acc
    {
        return _J_P_Biasa;
    }
    inline Eigen::Matrix3d getJVBiasg() const     // velocity / gyro
    {
        return _J_V_Biasg;
    }
    inline Eigen::Matrix3d getJVBiasa() const     // velocity / acc
    {
        return _J_V_Biasa;
    }
    inline Eigen::Matrix3d getJRBiasg() const     // rotation / gyro
    {
        return _J_R_Biasg;
    }
    // noise covariance propagation of delta measurements
    // note: the order is rotation-velocity-position here
    inline Eigen::Matrix<double, 9, 9> getCovPVPhi() const {
        return _cov_P_V_Phi;
    }
    inline double getDeltaTime() const {
        return _delta_time;
    }

    // Lie Algebra Functions
    Eigen::Matrix3d RightJacobianSO3(const Eigen::Vector3d &v);
    Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d &v);
    Eigen::Matrix3d NormalizeRotation(const Eigen::Matrix3d &R);

    /*
     * NOTE:
     * don't add pointer as member variable.
     * operator = is used in g2o
    */
    // delta measurements, position/velocity/rotation(matrix)
    Eigen::Vector3d _delta_P;    // P_k+1 = P_k + V_k*dt + R_k*a_k*dt*dt/2
    Eigen::Vector3d _delta_V;    // V_k+1 = V_k + R_k*a_k*dt
    Eigen::Matrix3d _delta_R;    // R_k+1 = R_k*exp(w_k*dt).     note: Rwc, Rwc'=Rwc*[w_body]x
    // jacobian of delta measurements w.r.t bias of gyro/acc
    Eigen::Matrix3d _J_P_Biasg;     // position / gyro
    Eigen::Matrix3d _J_P_Biasa;     // position / acc
    Eigen::Matrix3d _J_V_Biasg;     // velocity / gyro
    Eigen::Matrix3d _J_V_Biasa;     // velocity / acc
    Eigen::Matrix3d _J_R_Biasg;   // rotation / gyro
    // noise covariance propagation of delta measurements
    Eigen::Matrix<double, 9, 9> _cov_P_V_Phi;
    double _delta_time;

    // covariance of measurement
    static Eigen::Matrix3d GyrMeasCov;        
    static Eigen::Matrix3d AccMeasCov;        

    // covariance of bias random walk, RW stands for random walk
    static Eigen::Matrix3d GyrBiasRWCov;     
    static Eigen::Matrix3d AccBiasRWCov;    
};

} // namespace IMU

} // namespace demoam

#endif // DEMOAM__IMU_TYPES_H
