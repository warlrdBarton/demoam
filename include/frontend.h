#ifndef DEMOAM__FRONTEND_H
#define DEMOAM__FRONTEND_H

#include "common_include.h"
#include "imu_types.h"

namespace demoam {

class Frame;
class Camera;
class Map;
class Backend;
class Viewer;

enum FrontendStatus { 
    INITING, 
    OK, 
    RECENTLY_LOST, 
    LOST 
};

class Frontend {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW 
    Frontend();
    bool AddFrame(std::shared_ptr<Frame> frame);
    void Stop();
    
    void SetMap(std::shared_ptr<Map> map) { map_ = map; }
    void SetBackend(std::shared_ptr<Backend> backend) { backend_ = backend; }
    void SetViewer(std::shared_ptr<Viewer> viewer) { viewer_ = viewer; }
    void SetCameras(std::shared_ptr<Camera> left, std::shared_ptr<Camera> right) {
        camera_left_ = left;
        camera_right_ = right;
    }    
    FrontendStatus GetStatus() const { return status_; }
 
 private:
    bool StereoInit();
    bool BuildInitMap();
    bool Track();
    bool Reset();
    int DetectFastInLeft();
    int SearchLastFrameOpticalFlow();
    int SearchInRightOpticalFlow();
    int EstimateCurrentPose();
    bool InsertKeyFrame();
    void SetObservationsForKeyFrame();
    int TriangulateNewPoints();
    void SaveTrajectoryKITTI();
    int TrackLocalMap();
    void PredictCurrentPose();


    std::ofstream save_to_file_;

    // Data
    FrontendStatus status_ = FrontendStatus::INITING;
    
    std::shared_ptr<Frame> current_frame_ = nullptr;
    std::shared_ptr<Frame> last_frame_ = nullptr;
    std::shared_ptr<Frame> last_keyframe_ = nullptr;
    std::shared_ptr<Camera> camera_left_ = nullptr;
    std::shared_ptr<Camera> camera_right_ = nullptr;

    std::shared_ptr<Map> map_ = nullptr;
    std::shared_ptr<Backend> backend_ = nullptr;
    std::shared_ptr<Viewer> viewer_ = nullptr;

    Sophus::SE3d relative_motion_;
    std::vector<IMU::Point, Eigen::aligned_allocator<IMU::Point>> imu_meas_since_last_KF_;

    int tracking_inliers_ = 0;

    int num_features_init_ = 100;
    int num_features_tracking_good_ = 50;
    int num_features_needed_for_keyframe_ = 80;

    std::shared_ptr<cv::FastFeatureDetector> fast_detector_;
};

} // namespace demoam

#endif // DEMOAM__FRONTEND_H