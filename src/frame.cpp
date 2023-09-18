#include "frame.h"

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
    imu_preintegrator_from_RefKF_.reset();
    imu_preintegrator_from_RefKF_ = std::make_shared<IMUPreintegration>(this->BiasG(), this->BiasA());

    {   // consider the gap between the last KF and the first IMU
        // delta time
        double dt = std::max(0., imu_meas_[0].timestamp_ - reference_KF_.lock()->time_stamp_);
        // update pre-integrator
        imu_preintegrator_from_RefKF_->Integrate(imu_meas_[0], dt);
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
        imu_preintegrator_from_RefKF_->Integrate(imu_meas_[i], dt);
    }
}

} // namespace demoam