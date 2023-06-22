#ifndef DEMOAM__FRAME_H
#define DEMOAM__FRAME_H

#include "common_include.h"

namespace demoam {
    
struct Feature;
struct MapPoint;

struct Frame {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  
    static std::shared_ptr<Frame> CreateFrame();
    Frame(){}
    void SetKeyFrame();
    Sophus::SE3d Pose() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return pose_;
    }
    void SetPose(const Sophus::SE3d& pose) {
        std::unique_lock<std::mutex> lck(data_mutex_);
        pose_ = pose;
    }

    unsigned long id_ = 0;
    unsigned long keyframe_id_ = 0;
    bool is_keyframe_ = false;
    double time_stamp_;
    Sophus::SE3d pose_;
    std::mutex data_mutex_;
    cv::Mat img_left_, img_right_;

    std::vector<std::shared_ptr<Feature>> features_left_;
    std::vector<std::shared_ptr<Feature>> features_right_;

};

}  // namespace demoam

#endif  // DEMOAM__FRAME_H