#include "camera.h"

namespace demoam {
Eigen::Vector3d Camera::world2camera(const Eigen::Vector3d& pw, const Sophus::SE3d& tcw) {
    return pose_ * tcw * pw;
}

Eigen::Vector3d Camera::camera2world(const Eigen::Vector3d& pc, const Sophus::SE3d& tcw) {
    return tcw.inverse() * pose_inv_ * pc;
}

Eigen::Vector2d  Camera::camera2pixel(const Eigen::Vector3d& pc) {
    return Eigen::Vector2d(fx_ * pc(0, 0) / pc(2, 0) + cx_, 
                           fy_ * pc(1, 0) / pc(2, 0) + cy_);
}

Eigen::Vector3d Camera::pixel2camera(const Eigen::Vector2d& pp, double z) {
    return Eigen::Vector3d((pp(0, 0) - cx_) * z / fx_,
                           (pp(1, 0) - cy_) * z / fy_,
                           z);
}

Eigen::Vector3d Camera::pixel2world(const Eigen::Vector2d& pp, const Sophus::SE3d& tcw, double z) {
    return camera2world(pixel2camera(pp, z), tcw);     
}

Eigen::Vector2d Camera::world2pixel(const Eigen::Vector3d& pw, const Sophus::SE3d& tcw) {
    return camera2pixel(world2camera(pw, tcw));
}
}