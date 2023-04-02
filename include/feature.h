#pragma once

#include "common_include.h"

namespace demoam {
struct Frame;
struct MapPoint;

struct Feature {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Feature(){}
    Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint& kp)
        : position_(kp), frame_(frame) {     
    }

    cv::KeyPoint position_;
    std::weak_ptr<Frame> frame_;
    std::weak_ptr<MapPoint> mappoint_;
    bool is_outlier_ = false;
    bool is_on_left_image_ = true;
};
}