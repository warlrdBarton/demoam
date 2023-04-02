#pragma once

#include "common_include.h"

namespace demoam {
class Frame; 
class Camera;

class Dataset {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Dataset(const std::string& dataset_path);
    bool Init();
    std::shared_ptr<Frame> NextFrame();
    std::shared_ptr<Camera> GetCamera(int camera_id) const {
        return cameras_[camera_id];
    }
    
 private:
    std::string dataset_path_;
    int current_image_index_ = 0;
    std::vector<std::shared_ptr<Camera>> cameras_;    
};
}