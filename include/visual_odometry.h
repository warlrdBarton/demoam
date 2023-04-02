#pragma once

#include "common_include.h"

namespace demoam {
class Dataset; class Frontend; class Backend; class Map; class Viewer;

class VisualOdometry {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW; 
    VisualOdometry(std::string config_path);
    bool Init();
    void Run();
    bool Step();
    
 private:
    std::string config_file_path_;
    std::shared_ptr<demoam::Dataset> dataset_ = nullptr;
    std::shared_ptr<demoam::Frontend> frontend_ = nullptr;
    std::shared_ptr<demoam::Backend> backend_ = nullptr;
    std::shared_ptr<demoam::Map> map_ = nullptr;
    std::shared_ptr<demoam::Viewer> viewer_ = nullptr;
        
};
}