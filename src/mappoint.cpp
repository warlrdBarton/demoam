#include "mappoint.h"

#include "feature.h"

namespace demoam {
MapPoint::MapPoint(long id, Eigen::Vector3d position) : id_(id), pos_(position) {}

std::shared_ptr<MapPoint> MapPoint::CreateNewMapPoint() {
    static long factory_id = 0;
    std::shared_ptr<MapPoint> new_mappoint(new MapPoint);
    new_mappoint -> id_ = factory_id++;
    return new_mappoint;
}

void MapPoint::RemoveObservation(std::shared_ptr<Feature> feature) {
    std::unique_lock<std::mutex> lck(data_mutex_);
    for (auto iter = observations_.begin(); iter != observations_.end(); ++iter) {
        if (iter -> lock() == feature) { // Creates a new std::shared_ptr that shares ownership of the managed object. If there is no managed object, i.e. *this is empty, then the returned shared_ptr also is empty.
            observations_.erase(iter);
            feature -> mappoint_.reset();
            observed_times_--;
            break;
        }
    }
}

} // namespace demoam