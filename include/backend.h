#ifndef DEMOAM__BACKEND_H   
#define DEMOAM__BACKEND_H

#include "common_include.h"

namespace demoam {

class Map;
class Camera;
class Frame;
class MapPoint;

class Backend {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Backend();
    void SetMap(std::shared_ptr<Map> map) { map_ = map; }
    void SetCameras(std::shared_ptr<Camera> left, std::shared_ptr<Camera> right) {
        camera_left_ = left;
        camera_right_ = right;
    }
    void SetGravity(const Eigen::Vector3d& gravity) {
        g_ = gravity;
    }

    void UpdateMap();
    void Stop();

    bool IsCurrentlyBusy();

 private:
    void BackendLoop();
    void Optimize(std::unordered_map<u_long, std::shared_ptr<Frame>>& keyframes, std::unordered_map<u_long, std::shared_ptr<MapPoint>>& mappoints);
    void LocalBundleAdjustment(std::unordered_map<u_long, std::shared_ptr<Frame>>& keyframes, std::unordered_map<u_long, std::shared_ptr<MapPoint>>& mappoints);
    void LocalInertialBA(std::unordered_map<u_long, std::shared_ptr<Frame>>& keyframes, std::unordered_map<u_long, std::shared_ptr<MapPoint>>& mappoints);

    std::shared_ptr<Map> map_ = nullptr;
    std::shared_ptr<Camera> camera_left_ = nullptr;
    std::shared_ptr<Camera> camera_right_ = nullptr;

    std::mutex data_mutex_;

    std::thread backend_thread_;
    std::condition_variable map_update_;
    std::atomic_bool backend_running_;

    bool is_busy_ = false;
    Eigen::Vector3d g_ = Eigen::Vector3d::Zero();
};

}  // namespace demoam

#endif  // DEMOAM__BACKEND_H