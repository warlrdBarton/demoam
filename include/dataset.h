#ifndef DEMOAM__DATASET_H
#define DEMOAM__DATASET_H

#include "common_include.h"

namespace demoam {

class Frame; 
class Camera;

class Dataset {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW    
    Dataset(const std::string& dataset_path, const std::string& imu_path);
    bool Init();
    std::shared_ptr<Frame> NextFrame();
    std::shared_ptr<Camera> GetCamera(int camera_id) const {
        return cameras_[camera_id];
    }
    
 private:
    bool LoadCalib();
    bool LoadImages();
    bool LoadIMU();
    int first_imu_ = 0;
    std::string dataset_path_, imu_path_;
    std::vector<cv::Point3f> vAcc_, vGyro_;
    std::vector<double> vTimestampsCam_, vTimestampsImu_;
    int current_image_index_ = 0;
    std::vector<std::shared_ptr<Camera>> cameras_;    
};

} // namespace demoam

#endif // DEMOAM__DATASET_H