#include "map.h"
#include "frame.h"
#include "mappoint.h"
#include "feature.h"

namespace demoam {
void Map::InsertKeyFrame(std::shared_ptr<Frame> frame) {
    current_frame_ = frame;
    keyframes_[frame -> keyframe_id_] = frame;
    active_keyframes_[frame -> keyframe_id_] = frame;
    if (active_keyframes_.size() > num_active_keyframes_) {
        RemoveOldKeyFrame();
    }
}

void Map::InsertMapPoint(std::shared_ptr<MapPoint> mappoint) {
    mappoints_[mappoint -> id_] = mappoint;
    active_mappoints_[mappoint -> id_] = mappoint;
}

void Map::RemoveOldKeyFrame() {
    // TODO:
    // easily remove the oldest in time;
    if (current_frame_ == nullptr) return;
    double max_dis = 0, min_dis = 9999;
    double max_kf_id = 0, min_kf_id = 0;
    auto Twc = current_frame_ -> Pose().inverse();
    for (auto& kf : active_keyframes_) {
        if (kf.second == current_frame_) continue;
        auto dis = (kf.second -> Pose() * Twc).log().norm();
        if (dis > max_dis) {
            max_dis = dis;
            max_kf_id = kf.first;
        }
        if (dis < min_dis) {
            min_dis = dis;
            min_kf_id = kf.first;
        }
    }

    const double min_dis_th = 0.2;
    std::shared_ptr<Frame> frame_to_remove = nullptr;
    if (min_dis < min_dis_th) {
        frame_to_remove = keyframes_.at(min_kf_id);
    } else {
        frame_to_remove = keyframes_.at(max_kf_id);
    }

    LOG(INFO) << "Map::RemoveOldKeyFrame(): Remove keyframe " << frame_to_remove -> keyframe_id_;
    active_keyframes_.erase(frame_to_remove -> keyframe_id_);
    for (auto feat : frame_to_remove -> features_left_) {
        auto mp = feat -> mappoint_.lock();
        if (mp) {
            mp -> RemoveObservation(feat);
        }     
    }
    for (auto feat : frame_to_remove -> features_right_) {
        if (!feat) continue;
        auto mp = feat -> mappoint_.lock();
        if (mp) {
            mp -> RemoveObservation(feat);
        }
    }
    
    CleanMap();
}

void Map::CleanMap() {
    int cnt_mappoint_removed = 0;
    for (auto it = active_mappoints_.begin(); it != active_mappoints_.end();) {
        if (it -> second -> observed_times_ == 0) {
            it = active_mappoints_.erase(it);
            cnt_mappoint_removed++;
        } else {
            ++it;
        }
    }
    LOG(INFO) << "Map::CleanMap(): Removed " << cnt_mappoint_removed << " active mappoints.";
}
}
