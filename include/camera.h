#ifndef DEMOAM__CAMERA_H
#define DEMOAM__CAMERA_H

#include "common_include.h"

namespace demoam {

class Camera {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   
    Camera(){}
    Camera(double fx, double fy, double cx, double cy, double baseline,
           const Sophus::SE3d& pose)
        : fx_(fx), fy_(fy), cx_(cx), cy_(cy), baseline_(baseline), pose_(pose) {
        pose_inv_ = pose_.inverse();
    }

    Eigen::Matrix3d K() const {
        Eigen::Matrix3d k;
        k << fx_, 0, cx_,
             0, fy_, cy_,
             0, 0, 1;
        return k; 
    }
    Sophus::SE3d Pose() const {
        return pose_;
    }

    Eigen::Vector3d world2camera(const Eigen::Vector3d& pw, const Sophus::SE3d& tcw);

    Eigen::Vector3d camera2world(const Eigen::Vector3d& pc, const Sophus::SE3d& tcw);

    Eigen::Vector2d camera2pixel(const Eigen::Vector3d& pc);

    Eigen::Vector3d pixel2camera(const Eigen::Vector2d& pp, double z = 1);

    Eigen::Vector3d pixel2world(const Eigen::Vector2d& pp, const Sophus::SE3d& tcw, double z = 1);

    Eigen::Vector2d world2pixel(const Eigen::Vector3d& pw, const Sophus::SE3d& tcw);


 private:
    double fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0, baseline_ = 0;
    Sophus::SE3d pose_; //Tcc0
    Sophus::SE3d pose_inv_;
};

} // namespace demoam

#endif // DEMOAM__CAMERA_H