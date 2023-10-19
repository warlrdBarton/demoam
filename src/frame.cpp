#include "frame.h"
#include "feature.h"
#include "mappoint.h"
#include "camera.h"

namespace demoam {
    
std::shared_ptr<Frame> Frame::CreateFrame() {
    static long factory_id = 0;
    std::shared_ptr<Frame> new_frame(new Frame);
    new_frame -> id_ = factory_id++;
    return new_frame;
}

void Frame::SetKeyFrame() {
    static long keyframe_factory_id = 0;
    is_keyframe_ = true;
    keyframe_id_ = keyframe_factory_id++;
}

void Frame::ReComputeIMUPreIntegration() {
    if (reference_KF_.expired()) return;
    if (imu_preintegrator_from_RefKF_ != nullptr && reference_KF_.lock()->is_bias_updated_recently_ == false) return; // bias updated, no need to recompute;
    imu_preintegrator_from_RefKF_->reset();

    Vector3d bg = reference_KF_.lock()->BiasG();
    Vector3d ba = reference_KF_.lock()->BiasA();

    {   // consider the gap between the last KF and the first IMU
        // delta time
        double dt = std::max(0., imu_meas_[0].timestamp_ - reference_KF_.lock()->time_stamp_);
        // update pre-integrator
        imu_preintegrator_from_RefKF_->update(imu_meas_[0].gyro_ - bg, imu_meas_[0].acce_ - ba, dt);
    }
    for (size_t i = 0; i < imu_meas_.size(); i++) {
        double nextt;
        if (i == imu_meas_.size() - 1)
            nextt = this->time_stamp_;
        else
            nextt = imu_meas_[i+1].timestamp_;  // regular condition, next is imu data
        // delta time
        double dt = std::max(0., nextt - imu_meas_[i].timestamp_);
        // update pre-integrator
        imu_preintegrator_from_RefKF_->update(imu_meas_[i].gyro_ - bg, imu_meas_[i].acce_ - ba, dt);
    }

    reference_KF_.lock()->is_bias_updated_recently_ = false; // imu preintegration with new bias is done 
}

int Frame::TrackedMapPoints(const int &minObs) {
    int nPoints = 0;
    const bool bCheckObs = minObs > 0;
    std::unique_lock<std::mutex> lock(data_mutex_);
    int N = features_left_.size();
    for (int i = 0; i < N; i++) {
        if (features_left_[i] == nullptr)
            continue;
        auto pMP = (features_left_[i]->mappoint_).lock();
        if (pMP) {
            if (!pMP->is_outlier_) {
                if (bCheckObs) {
                    if (pMP->observed_times_ >= minObs)
                        nPoints++;
                } else
                    nPoints++;
            }
        }
    }
    return nPoints;
}

bool Frame::isInFrustum(std::shared_ptr<MapPoint> pMP, std::shared_ptr<Camera> pCam, float viewingCosLimit, int boarder = 20) {
    auto pw = pMP->Pos();
    const Vector3d pc = pCam->world2camera(pw, this->Pose());
    const float pcx = pc[0], pcy = pc[1], pcz = pc[2];
    
    // Checking depth
    if (pcz < settings::minPointDis || pcz > settings::maxPointDis) {
        return false;
    }

    auto px = pCam->camera2pixel(pc);
    const float u = px[0], v = px[1];
    int imageWidth = img_left_.cols, imageHeight = img_left_.rows;
    if (u < boarder || u > (imageWidth - boarder))
        return false;
    if (v < boarder || v > (imageHeight - boarder))
        return false;

/*     auto twc = (this->Pose()).inverse().translation().matrix();
    const float dist = (pw - twc).norm(); */

    return true;
}

} // namespace demoam