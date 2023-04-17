#pragma once

#include "common_include.h"

namespace demoam {
class Frame;
class Camera;

class FeatureTracker {
 public:
    static int DetectFastInLeft(std::shared_ptr<Frame> frame);
    static int SearchLastFrameOpticalFlow(std::shared_ptr<Frame> last_frame, std::shared_ptr<Frame> current_frame, std::shared_ptr<demoam::Camera> camera_left, int f_threshold = 1);
    static int SearchInRightOpticalFlow(std::shared_ptr<Frame> current_frame, std::shared_ptr<demoam::Camera> camera_right, int f_threshold = 1);

 private:
    static std::shared_ptr<cv::FastFeatureDetector> fast_detector_;
};
}