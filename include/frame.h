#ifndef DEMOAM__FRAME_H
#define DEMOAM__FRAME_H

#include "common_include.h"
#include "imu_types.h"
#include "config.h"

namespace demoam {
    
struct Feature;
struct MapPoint;
struct Camera;

struct Frame {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  

    static std::shared_ptr<Frame> CreateFrame();

    Frame(){}

    void SetKeyFrame();

    void ReComputeIMUPreIntegration();

    // --------------------------------------------------------------------
    // Getters 
    // Tcw
    inline Sophus::SE3d Pose() { 
        std::unique_lock<std::mutex> lck(data_mutex_);
        return pose_;
    }

    //Twb
    inline Sophus::SE3d ImuPose() { 
        std::unique_lock<std::mutex> lck(data_mutex_);
        return pose_.inverse() * settings::Tcb;
    }

    inline void SetPose(const Sophus::SE3d& pose) {
        std::unique_lock<std::mutex> lck(data_mutex_);
        pose_ = pose;
    }

    inline void SetPoseFromIMU(const Sophus::SE3d& pose) {
        std::unique_lock<std::mutex> lck(data_mutex_);
        SE3d Twb = pose;
        SE3d Twc = Twb * settings::Tbc;
        SE3d Tcw = Twc.inverse();
        pose_ = Tcw;
    }

    inline Eigen::Vector3d Velocity() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return velocity_n_bias_.segment<3>(0);
    }

    inline Eigen::Vector3d BiasG() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return velocity_n_bias_.segment<3>(3);
    }    

    inline Eigen::Vector3d BiasA() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return velocity_n_bias_.segment<3>(6);
    }

    inline void SetVelocitynBias(const Eigen::Vector3d &velocity, const Eigen::Vector3d &bias_g, const Eigen::Vector3d &bias_a) {
        std::unique_lock<std::mutex> lck(data_mutex_);
        velocity_n_bias_.segment<3>(0) = velocity;
        velocity_n_bias_.segment<3>(3) = bias_g;
        velocity_n_bias_.segment<3>(6) = bias_a;
        is_bias_updated_recently_ = true;
    }

    inline void SetVelocity(const Eigen::Vector3d &velocity) {
        std::unique_lock<std::mutex> lck(data_mutex_);
        velocity_n_bias_.segment<3>(0) = velocity;
    }

    inline void SetBiasG(const Vector3d &bias_g) {
        std::unique_lock<std::mutex> lck(data_mutex_);
        velocity_n_bias_.segment<3>(3) = bias_g;
        is_bias_updated_recently_ = true;
    }

    inline void SetBiasA(const Eigen::Vector3d &bias_a) {
        std::unique_lock<std::mutex> lck(data_mutex_);
        velocity_n_bias_.segment<3>(6) = bias_a;
        is_bias_updated_recently_ = true;
    }

    const auto& GetIMUPreInt() { return imu_preintegrator_from_RefKF_; }

    // count the well tracked map points in this frame
    int TrackedMapPoints(const int &minObs);

    bool isInFrustum(std::shared_ptr<MapPoint> pMP, std::shared_ptr<Camera> pCam, float viewingCosLimit, int boarder);
 
    // --------------------------------------------------------------------
    // Data
    double time_stamp_;

    std::mutex data_mutex_;

    unsigned long id_ = 0;
    unsigned long keyframe_id_ = 0;

    bool is_keyframe_ = false;
    bool is_bias_updated_recently_ = true;

    std::weak_ptr<Frame> reference_KF_;
    
    Sophus::SE3d pose_;
    Eigen::Matrix<double, 9, 1> velocity_n_bias_ = Eigen::Matrix<double, 9, 1>::Zero();

    std::vector<IMU, Eigen::aligned_allocator<IMU>> imu_meas_;  // For KF, it stores measurements from reference KeyFrame
                                                                            // For the rest frames, keeps measurements up to the last frame

    std::shared_ptr<IMUPreIntegration> imu_preintegrator_from_RefKF_; // this one trace back to last KF,
                                                 // inter-frame IMU integration use a temporary one stored in Frontend Class


    cv::Mat img_left_, img_right_;

    std::vector<std::shared_ptr<Feature>> features_left_;
    std::vector<std::shared_ptr<Feature>> features_right_;

};

}  // namespace demoam

#endif  // DEMOAM__FRAME_H