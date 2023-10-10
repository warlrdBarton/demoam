#ifndef DEMOAM__MAP_H
#define DEMOAM__MAP_H

#include "common_include.h"

namespace demoam {
class Frame;
class MapPoint;

class Map {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW 
    Map(){}
    void InsertKeyFrame(std::shared_ptr<Frame> frame);
    void InsertMapPoint(std::shared_ptr<MapPoint> mappoint);

    std::unordered_map<u_long, std::shared_ptr<MapPoint>> GetAllMapPoints() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return mappoints_;
    }
    std::unordered_map<u_long, std::shared_ptr<MapPoint>> GetActiveMapPoints() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_mappoints_;
    }
    std::unordered_map<u_long, std::shared_ptr<Frame>> GetAllKeyFrames() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return keyframes_;
    }
    std::unordered_map<u_long, std::shared_ptr<Frame>> GetActiveKeyFrames() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_keyframes_;
    }

    void CleanMap();

    bool isImuInitialized() {
        return is_imu_initialized_;
    }
    bool is_imu_initialized_ = false;
    
 private:
    void RemoveOldKeyFrame();
    
    std::mutex data_mutex_;
    std::unordered_map<u_long, std::shared_ptr<MapPoint>> mappoints_, active_mappoints_;
    std::unordered_map<u_long, std::shared_ptr<Frame>> keyframes_, active_keyframes_;
    
    std::shared_ptr<Frame> current_frame_ = nullptr;
    size_t num_active_keyframes_ = 10;


 
};

} // namespace demoam

#endif // DEMOAM__MAP_H