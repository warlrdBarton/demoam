#ifndef DEMOAM__VISUAL_ODOMETRY_H
#define DEMOAM__VISUAL_ODOMETRY_H

#include "common_include.h"

namespace demoam {

class Dataset; class Frontend; class Backend; class Map; class Viewer;

class VisualOdometry {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VisualOdometry(std::string config_path);
    bool Init();
    void Run();
    bool Step();
    
 private:
    std::string config_file_path_;
    std::shared_ptr<Dataset> dataset_ = nullptr;
    std::shared_ptr<Frontend> frontend_ = nullptr;
    std::shared_ptr<Backend> backend_ = nullptr;
    std::shared_ptr<Map> map_ = nullptr;
    std::shared_ptr<Viewer> viewer_ = nullptr;
};

} // namespace demoam

#endif // DEMOAM__VISUAL_ODOMETRY_H