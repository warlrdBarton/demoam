#ifndef DEMOAM__MAPPOINT_H
#define DEMOAM__MAPPOINT_H

#include "common_include.h"

namespace demoam {

struct Frame;
struct Feature;

struct MapPoint {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  
    static std::shared_ptr<MapPoint> CreateNewMapPoint();
    MapPoint() {}
    MapPoint(long id, Eigen::Vector3d position);

    Eigen::Vector3d Pos() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return pos_;
    }
    void SetPos(const Eigen::Vector3d& pos) {
        std::unique_lock<std::mutex> lck(data_mutex_);
        pos_ = pos;
    }

    std::list<std::weak_ptr<Feature>> Observations() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return observations_;
    }
    void AddObservation(std::shared_ptr<Feature> feature) {
        std::unique_lock<std::mutex> lck(data_mutex_);
        observations_.push_back(feature);
        observed_times_++;
    }
    void RemoveObservation(std::shared_ptr<Feature> feature);
    
    unsigned long id_ = 0;
    bool is_outlier_ = false;
    Eigen::Vector3d pos_ = Eigen::Vector3d::Zero();
    std::mutex  data_mutex_;
    int observed_times_ = 0;
    std::list<std::weak_ptr<Feature>> observations_;
};

} // namespace demoam

#endif // DEMOAM__MAPPOINT_H