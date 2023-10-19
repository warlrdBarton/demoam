#include "frontend.h"

#include <glog/logging.h>
#include "geometric_tools.h"
#include "backend.h"
#include "frame.h"
#include "map.h"
#include "mappoint.h"
#include "feature.h"
#include "camera.h"
#include "config.h"
#include "g2o_types.h"
#include "feature.h"
#include "viewer.h"

namespace demoam {
Frontend::Frontend() {
    fast_detector_ = cv::FastFeatureDetector::create(Config::Get<int>("threshold_fast_detector"), true);
    
    imu_preintegrator_from_RefKF_ = std::make_shared<IMUPreIntegration>();
    num_features_init_ = Config::Get<int>("num_features_init");
    num_features_tracking_good_ = Config::Get<int>("num_features_tracking_good");
    num_features_needed_for_keyframe_ = Config::Get<int>("num_features_needed_for_keyframe");

    num_kfs_for_imu_init_ = Config::Get<int>("NUM_KFs_FOR_IMU_INIT");

    ratio_of_tracked_mp_for_new_kfs_ = Config::Get<float>("RATIO_OF_TRACKED_MP_FOR_NEW_KFS");

    save_to_file_.open("./traj.txt", std::ios::trunc);
}

void Frontend::Stop() {
    save_to_file_.close();
}

bool Frontend::AddFrame(std::shared_ptr<Frame> frame) {
    LOG(INFO) << "--------------------------------------------------------------------------------";
    LOG(INFO) << "------------------------------Status:"<< status_ << "----------------------------------------------";
    LOG(INFO) << "--------------------------------------------------------------------------------";
    LOG(INFO) << "------------------------------Frame:" << frame->id_ << "----------------------------------------";
    LOG(INFO) << "--------------------------------------------------------------------------------";

    current_frame_ = frame;

    PreintegrateIMU();
    
    if (last_keyframe_) 
        current_frame_->reference_KF_ = last_keyframe_;

    switch (status_) {
        case FrontendStatus::INITING:
            StereoInit();
            break;
        case FrontendStatus::OK:
        case FrontendStatus::RECENTLY_LOST:
            Track();
            break;
        case FrontendStatus::LOST:
            Reset();
            break;
    }
    SaveTrajectoryKITTI();

    if (status_ == FrontendStatus::OK) {
        last_frame_ = current_frame_;
    }

    //if (map_->isImuInitialized()) cv::waitKey(0);

    return status_ != FrontendStatus::LOST;
    //return true;
}

bool Frontend::StereoInit() {
    DetectFastInLeft();
    if (SearchInRightOpticalFlow() < num_features_init_) {
        LOG(INFO) << "Frontend::StereoInit(): Stereo init failed.";
        return false;
    } 
    bool build_map_success = BuildInitMap();
    if (build_map_success) {
        if (viewer_) {
            viewer_ -> AddCurrentFrame(current_frame_);
            viewer_ -> UpdateMap();
        }
        status_ = FrontendStatus::OK;
        last_keyframe_ = current_frame_;
        return true;
    }
    return false;       
}

bool Frontend::BuildInitMap() {
    int cnt_init_mappoints = 0;
    VecSE3d poses{camera_left_ -> Pose(), camera_right_ -> Pose()}; 
    for (size_t i = 0; i < current_frame_ -> features_left_.size(); ++i) {
        if (current_frame_ -> features_right_[i] == nullptr) continue;
        VecVector3d points{
            camera_left_ -> pixel2camera(
                Eigen::Vector2d(current_frame_->features_left_[i] -> position_.pt.x,
                                current_frame_->features_left_[i] -> position_.pt.y)),
            camera_right_ -> pixel2camera(
                Eigen::Vector2d(current_frame_->features_right_[i] -> position_.pt.x,
                                current_frame_->features_right_[i] -> position_.pt.y))
        };
        Eigen::Vector3d pw = Eigen::Vector3d::Zero();
        if (TriangulatePoints(poses, points, pw)) { // https://www.cvlibs.net/datasets/kitti/setup.php
            auto new_mappoint = MapPoint::CreateNewMapPoint();
            new_mappoint -> SetPos(pw);
            new_mappoint -> AddObservation(current_frame_ -> features_left_[i]);
            new_mappoint -> AddObservation(current_frame_ -> features_right_[i]);
            current_frame_ -> features_left_[i] -> mappoint_ = new_mappoint;
            current_frame_ -> features_right_[i] -> mappoint_ = new_mappoint;
            cnt_init_mappoints++;
            map_ -> InsertMapPoint(new_mappoint);
        }
    }
    current_frame_ -> SetKeyFrame();
    map_ -> InsertKeyFrame(current_frame_);
    backend_ -> UpdateMap();

    LOG(INFO) << "Frontend::BuildInitMap(): Initial map was sucessfully built with " << cnt_init_mappoints << " mappoints";
    return true;
}

bool Frontend::Track() {    
    PredictCurrentPose();

    if (status_ == FrontendStatus::OK) {
        SearchLastFrameOpticalFlow();
        tracking_inliers_ = OptimizeCurrentPose();
    }
    // if tracking lastframe failed, try to track reference KF
    if (tracking_inliers_ < num_features_tracking_good_ || status_ == FrontendStatus::RECENTLY_LOST) {
        // Firstly, clear all features tracked from last frame in the above step;
        current_frame_->features_left_.clear();

        SearchReferenceKFOpticalFlow();
        tracking_inliers_ = OptimizeCurrentPose();
    }
    
    tracking_inliers_ = TrackLocalMap();
    
    if (tracking_inliers_ > num_features_tracking_good_) {
        status_  = FrontendStatus::OK;
    } else if (map_->isImuInitialized() && current_frame_->time_stamp_ - last_frame_->time_stamp_ < 5.0) { // IMU-Only Mode lasts 5 secs at most
        status_ = FrontendStatus::RECENTLY_LOST;
    } else {
        status_ = FrontendStatus::LOST;
    }
    
    InsertKeyFrame();

    if (map_->isImuInitialized() == false) {
        IMUInitialization();
    } 

    if (status_ == FrontendStatus::OK) relative_motion_ = current_frame_ -> Pose() * last_frame_ -> Pose().inverse();
    
    if (viewer_) viewer_ -> AddCurrentFrame(current_frame_);

    return true; 
}

int Frontend::DetectFastInLeft() {
    if (fast_detector_ == nullptr) {
        fast_detector_ = cv::FastFeatureDetector::create(10, true);
    }
    cv::Mat mask(current_frame_ -> img_left_.size(), CV_8UC1, 255);
    for (auto& feat : current_frame_ -> features_left_) {
        cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
                      feat->position_.pt + cv::Point2f(10, 10), 0, cv::FILLED);
    }
    std::vector<cv::KeyPoint> keypoints;
    fast_detector_ -> detect(current_frame_ -> img_left_, keypoints, mask);
    int cntDetected = 0;
    for (auto& kp : keypoints) {
        current_frame_ -> features_left_.push_back(
            std::shared_ptr<Feature>(new Feature(current_frame_, kp))
        );
        cntDetected++;
    }
    LOG(INFO) << "Frontend::DetectFastInLeft(): Keypoints detected in Left: " << cntDetected;
    return cntDetected;
}

int Frontend::SearchLastFrameOpticalFlow() {
    std::vector<cv::Point2f> kps_last, kps_current; 
    for (auto& kp : last_frame_ -> features_left_) {
        kps_last.push_back(kp -> position_.pt);
        auto mp = kp -> mappoint_.lock();
        if (mp) {
            auto px = camera_left_ -> world2pixel(mp -> pos_, current_frame_ -> Pose());
            kps_current.push_back(cv::Point2f(px[0], px[1]));
        } else {
            kps_current.push_back(kp -> position_.pt);
        }
    }
    std::vector<uchar> status;
    std::vector<float> error;
    cv::calcOpticalFlowPyrLK(last_frame_ -> img_left_, current_frame_ -> img_left_, 
                             kps_last, kps_current, status, error, cv::Size(11, 11), 1
    );
    //cv::findFundamentalMat(kps_current, kps_last, cv::FM_RANSAC, f_threshold, 0.99, status);
   
    int num_good_pts = 0;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            cv::KeyPoint kp(kps_current[i], 7);
            std::shared_ptr<Feature> new_feature(new Feature(current_frame_, kp));
            new_feature -> mappoint_ = last_frame_ -> features_left_[i] -> mappoint_;
            current_frame_ -> features_left_.push_back(new_feature);
            num_good_pts++;
        }
    }
    LOG(INFO) << "Frontend::TrackLastFrameOpticalFlow: " << num_good_pts << " keypoints tracked successfuly from LastFrame";
    return num_good_pts;
}

int Frontend::SearchInRightOpticalFlow() {
    std::vector<cv::Point2f> kps_left, kps_right;
    for (auto& kp : current_frame_ -> features_left_) {
        kps_left.push_back(kp -> position_.pt);
        auto mp = kp -> mappoint_.lock();
        if (mp) {
            auto px = camera_right_ -> world2pixel(mp -> pos_, current_frame_ -> Pose());
            kps_right.push_back(cv::Point2f(px[0], px[1]));
        } else {
            kps_right.push_back(kp -> position_.pt);
        }
    }
    std::vector<uchar> status;
    std::vector<float> error;
    cv::calcOpticalFlowPyrLK(current_frame_ -> img_left_, current_frame_ -> img_right_, 
                             kps_left, kps_right, status, error, cv::Size(11, 11), 1
    );
    //cv::findFundamentalMat(kps_right, kps_left, cv::FM_RANSAC, f_threshold, 0.99, status);
    int num_good_pts = 0;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            cv::KeyPoint kp(kps_right[i], 7);
            std::shared_ptr<Feature> new_feature(new Feature(current_frame_, kp));
            new_feature -> is_on_left_image_ = false;
            current_frame_ -> features_right_.push_back(new_feature);
            num_good_pts++;
        } else {
            current_frame_ -> features_right_.push_back(nullptr);
        }
    }
    LOG(INFO) << "Frontend::SearchInRightOpticalFlow: " << num_good_pts << " keypoints tracked successfuly in right image";
    return num_good_pts;
}

int Frontend::SearchReferenceKFOpticalFlow() {
    std::vector<cv::Point2f> kps_last, kps_current; 
    for (auto& kp : last_keyframe_ -> features_left_) {
        kps_last.push_back(kp -> position_.pt);
        auto mp = kp -> mappoint_.lock();
        if (mp) {
            auto px = camera_left_ -> world2pixel(mp -> pos_, current_frame_ -> Pose());
            kps_current.push_back(cv::Point2f(px[0], px[1]));
        } else {
            kps_current.push_back(kp -> position_.pt);
        }
    }
    std::vector<uchar> status;
    std::vector<float> error;
    cv::calcOpticalFlowPyrLK(last_keyframe_ -> img_left_, current_frame_ -> img_left_, 
                             kps_last, kps_current, status, error, cv::Size(21, 21), 3
    );
    //cv::findFundamentalMat(kps_current, kps_last, cv::FM_RANSAC, f_threshold, 0.99, status);
   
    int num_good_pts = 0;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            cv::KeyPoint kp(kps_current[i], 7);
            std::shared_ptr<Feature> new_feature(new Feature(current_frame_, kp));
            new_feature -> mappoint_ = last_keyframe_ -> features_left_[i] -> mappoint_;
            current_frame_ -> features_left_.push_back(new_feature);
            num_good_pts++;
        }
    }
    LOG(INFO) << "Frontend::SearchReferenceKFOpticalFlow: " << num_good_pts << " keypoints tracked successfuly from ReferenceKF";
    return num_good_pts;
}

int Frontend::OptimizeCurrentPose() {
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    VertexSE3Expmap *vertex_pose = new VertexSE3Expmap();
    vertex_pose -> setId(0);
    vertex_pose -> setEstimate(current_frame_ -> Pose());
    optimizer.addVertex(vertex_pose);

    Eigen::Matrix3d K = camera_left_ -> K();
    Sophus::SE3d left_extrinsics = camera_left_ -> Pose();

    int index = 0; 
    std::vector<EdgeSE3ProjectXYZPoseOnly *> edges;
    std::vector<std::shared_ptr<Feature>> features;
    for (auto& feat : (current_frame_ -> features_left_)) {
        auto mp = feat -> mappoint_.lock();
        if (mp) {
            features.push_back(feat);
            EdgeSE3ProjectXYZPoseOnly* edge = new EdgeSE3ProjectXYZPoseOnly(mp -> pos_, K, left_extrinsics);
            edge -> setId(index);
            edge -> setVertex(0, vertex_pose);
            edge -> setMeasurement(Eigen::Vector2d(feat -> position_.pt.x, feat -> position_.pt.y));
            edge -> setInformation(Eigen::Matrix2d::Identity());
            edge -> setRobustKernel(new g2o::RobustKernelHuber);
            edges.push_back(edge);
            optimizer.addEdge(edge);
            index++;
        }
    }
    
    const double chi2_th = 5.991; // chi2 val for Prob 0.05 with 2-Dof;
    int cnt_outlier = 0;
    for (int iter = 0; iter < 4; ++iter) {
        vertex_pose -> setEstimate(current_frame_->Pose()); // reset the pose in each round;
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        cnt_outlier = 0;

        for (size_t i = 0; i < edges.size(); ++i) {
            if (features[i] -> is_outlier_) {
                edges[i] -> computeError();
            }
            
            if (edges[i] -> chi2() > chi2_th) {
                features[i] -> is_outlier_ = true;
                edges[i] -> setLevel(1); // drop-out for the next round iter;
                cnt_outlier++;
            } else {
                features[i] -> is_outlier_ = false;
                edges[i] -> setLevel(0);
            }

            if (iter == 2) { // after 2 rounds, edges with great error shoule be sifted out; 
                edges[i] -> setRobustKernel(nullptr);
            }
        }
    } 
    LOG(INFO) << "Frontend::OptimizeCurrentPose(): Outlier/Inlier in pose estimating: " << cnt_outlier << "/" << features.size() - cnt_outlier;

    current_frame_ -> SetPose(vertex_pose -> estimate());

    // LOG(INFO) << "Current Pose = \n" << current_frame_ -> Pose().matrix();

    for (auto& feat : features) {
        if (feat -> is_outlier_) {
            feat -> mappoint_.reset(); // release the ref to the mappoint;
            feat -> is_outlier_ = false;
        }
    }
    return features.size() - cnt_outlier;
}

bool Frontend::InsertKeyFrame() {
    // Necessary Condition
    if (status_ != FrontendStatus::OK) {
        return false;
    }

    // Condition 0: tracking is weak
    bool c0 = tracking_inliers_ < num_features_needed_for_keyframe_;
    LOG(INFO) << "Frontend::InsertKeyFrame(): c0 : " << "num_features_needed_for_keyframe_ = " << num_features_needed_for_keyframe_ << " tracking_inliers_ = " << tracking_inliers_;

    // Condition 1: Few tracked points compared to reference keyframe.
    int nRefMatches = last_keyframe_->TrackedMapPoints(settings::minObsForGoodMapPoint); // matches in reference KeyFrame
    int curMatches = current_frame_->TrackedMapPoints(settings::minObsForGoodMapPoint); // matches in current KeyFrame
    bool c1 = (curMatches < nRefMatches * ratio_of_tracked_mp_for_new_kfs_);
    LOG(INFO) << "Frontend::InsertKeyFrame(): nRefMatches = " << nRefMatches << ", curMatches = " << curMatches;

    // TimeGap from last keyframe
    double timegap = Config::Get<double>("keyframeTimeGapTracking");
    bool cTimeGap = (current_frame_->time_stamp_ - last_keyframe_->time_stamp_) > timegap;

    LOG(INFO) << "Frontend::InsertKeyFrame(): timegap = " << timegap << ", cTimeGap = " << (current_frame_->time_stamp_ - last_keyframe_->time_stamp_);

    if (!(c0 || c1 || cTimeGap) || backend_->IsCurrentlyBusy()) {
        return false;
    }
    
    current_frame_ -> SetKeyFrame();
    current_frame_->imu_preintegrator_from_RefKF_ = imu_preintegrator_from_RefKF_;
    map_ -> InsertKeyFrame(current_frame_);

    LOG(INFO) << "Frontend::InsertKeyFrame(): Set frame " << current_frame_ -> id_ << " as keyframe " << current_frame_ -> keyframe_id_;

    SetObservationsForKeyFrame();
    DetectFastInLeft();
    SearchInRightOpticalFlow();
    TriangulateNewPoints();

    backend_-> UpdateMap();

    if (viewer_) viewer_ -> UpdateMap();

    last_keyframe_ = current_frame_;

    // Reset imu mearsurments
    imu_preintegrator_from_RefKF_ = std::make_shared<IMUPreIntegration>(); 
    imu_meas_since_RefKF_.clear();

    return true;
}

void Frontend::SetObservationsForKeyFrame() {
    for (auto &feat : current_frame_ -> features_left_) {
        auto mp = feat -> mappoint_.lock();
        if (mp) mp -> AddObservation(feat);
    }
}

int Frontend::TriangulateNewPoints() {
    int cnt_triangulated_pts = 0;
    Sophus::SE3d current_pose_Twc = current_frame_ -> Pose().inverse();
    VecSE3d poses{camera_left_ -> Pose(), camera_right_ -> Pose()}; 
    for (size_t i = 0; i < current_frame_ -> features_left_.size(); ++i) {
        if (current_frame_ -> features_left_[i] -> mappoint_.expired() == false
            || current_frame_ -> features_right_[i] == nullptr) {
                continue;
        }
        VecVector3d points{
            camera_left_ -> pixel2camera(
                Eigen::Vector2d(current_frame_->features_left_[i] -> position_.pt.x,
                                current_frame_->features_left_[i] -> position_.pt.y)),
            camera_right_ -> pixel2camera(
                Eigen::Vector2d(current_frame_->features_right_[i] -> position_.pt.x,
                                current_frame_->features_right_[i] -> position_.pt.y))
        };
        Eigen::Vector3d pw = Eigen::Vector3d::Zero();
        if (TriangulatePoints(poses, points, pw)) {
            auto new_mappoint = MapPoint::CreateNewMapPoint();
            pw = current_pose_Twc * pw;
            new_mappoint -> SetPos(pw);
            new_mappoint -> AddObservation(current_frame_ -> features_left_[i]);
            new_mappoint -> AddObservation(current_frame_ -> features_right_[i]);
            current_frame_ -> features_left_[i] -> mappoint_ = new_mappoint;
            current_frame_ -> features_right_[i] -> mappoint_ = new_mappoint;
            cnt_triangulated_pts++;
            map_ -> InsertMapPoint(new_mappoint);
        }
    }
    LOG(INFO) << "Frontend::TriangulateNewPoints(): KeyFrame provides " << cnt_triangulated_pts << " new mappoints.";
    return cnt_triangulated_pts; 
}

bool Frontend::Reset() {
    return true;
}

void Frontend::SaveTrajectoryKITTI() {
    Sophus::SE3d current_pose_Twc = current_frame_ -> Pose().inverse();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            save_to_file_ << std::setprecision(9) << current_pose_Twc.matrix()(i, j);
            if (!(i == 2 && j == 3)) save_to_file_ << " "; 
        }
    }
    save_to_file_ << "\n";
}

int Frontend::TrackLocalMap() {
    // TODO: TrackLocalMap
    return tracking_inliers_;
    /*

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    auto localmap = map_->GetActiveMapPoints();
    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    double timeCost = std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t1).count();
    LOG(INFO) << "Frontend::TrackLocalMap(): get local map points cost time: " << timeCost << endl;

    for (auto& [_, mp] : localmap) {
        if (mp) mp->is_track_in_view_ = false;
    }
    for (auto& feat: current_frame_->features_left_) {
        if (feat && !feat->mappoint_.expired() && feat->is_outlier_ == false)
            feat->mappoint_.lock()->is_track_in_view = true;
    }
    std::set<shared_ptr<MapPoint> > mpsInView;
    for (auto& [_, mp] : localmap) {
        if (mp && mp->is_outlier_ == false && mp->is_track_in_view == false && current_frame_->isInFrustum(mp, camera_left_, 0.5)) {
            mpsInView.insert(mp);
        }
    }
    if (mpsInView.empty())
        return tracking_inliers_;
    LOG(INFO) << "Call Search by direct projection" << endl;
    int cntMatches = mpMatcher->SearchByDirectProjection(current_frame_, mpsInView);
    LOG(INFO) << "Track local map matches: " << cntMatches << ", current features: " << current_frame_->features_left.size();
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    timeCost = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    LOG(INFO) << "Search local map points cost time: " << timeCost << endl;
    // Optimize Pose
    int optinliers = OptimizeCurrentPose();
    // Update MapPoints Statistics
    tracking_inliers_ = 0;
    for (shared_ptr<Feature> feat : current_frame_->mFeaturesLeft) {
        if (feat->mpPoint) {
            if (!feat->mbOutlier) {
                feat->mpPoint->IncreaseFound();
                if (feat->mpPoint->Status() == MapPoint::GOOD)
                    tracking_inliers_++;
            } else {
            }
        }
    }
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    timeCost = std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t1).count();
    LOG(INFO) << "Track local map cost time: " << timeCost << endl;
    LOG(INFO) << "Track Local map tracking_inliers_: " << tracking_inliers_ << endl;
    // Decide if the tracking was succesful
    if (tracking_inliers_ < setting::minTrackLocalMapInliers)
        return false;
    else
        return true;

    */
   

}

void Frontend::PredictCurrentPose() {
    if (map_->isImuInitialized() == false) {
        current_frame_ -> SetPose(relative_motion_ * last_frame_ -> Pose());
        return;
    }
    
    current_frame_->SetPose(last_keyframe_->Pose());
    current_frame_->SetVelocitynBias(last_keyframe_->Velocity(), last_keyframe_->BiasG(), last_keyframe_->BiasA());

    const auto& imupreint = imu_preintegrator_from_RefKF_;
    Matrix3d dR = imupreint->getDeltaR();
    Vector3d dP = imupreint->getDeltaP();
    Vector3d dV = imupreint->getDeltaV();
    double dt = imupreint->getDeltaTime();

    Vector3d Pwbpre = current_frame_->ImuPose().translation().matrix();     
    Matrix3d Rwbpre = current_frame_->ImuPose().rotationMatrix().matrix();
    Vector3d Vwbpre = current_frame_->Velocity();

    Matrix3d Rwb = Rwbpre * dR;
    Vector3d Pwb = Pwbpre + Vwbpre * dt + 0.5 * g_* dt * dt + Rwbpre * dP;
    Vector3d Vwb = Vwbpre + g_* dt + Rwbpre * dV;    
    
    SE3d TWC = SE3d(Rwb, Pwb) * settings::Tbc;
    SE3d TCW = TWC.inverse();

    current_frame_->SetPose(TCW);
    current_frame_->SetVelocity(Vwb);

    {
        if (viewer_) viewer_ -> AddCurrentFrame(current_frame_);
    }

}

void Frontend::PreintegrateIMU() { 
    if (last_keyframe_ == nullptr || last_frame_ == nullptr || current_frame_->imu_meas_.empty()) {
        return;
        // like the first frame 
    }
    Vector3d bg = last_keyframe_->BiasG();
    Vector3d ba = last_keyframe_->BiasA();

    imu_meas_since_RefKF_.insert(imu_meas_since_RefKF_.end(), 
                                   current_frame_->imu_meas_.begin(), 
                                   current_frame_->imu_meas_.end());

    std::vector<IMU, Eigen::aligned_allocator<IMU>>imu;
    
    // deal with the gap between the last KF(or lastframe) and the first IMU
    if (last_keyframe_->is_bias_updated_recently_ == true) { // recompute imu pre-integration if BIAS is updated recently
        imu_preintegrator_from_RefKF_->reset();
        imu = imu_meas_since_RefKF_;
        double dt = std::max(0., imu[0].timestamp_ - last_keyframe_->time_stamp_);
        imu_preintegrator_from_RefKF_->update(imu[0].gyro_ - bg, imu[0].acce_ - ba, dt);
    } else {
        imu = current_frame_->imu_meas_;
        double dt = std::max(0., imu[0].timestamp_ - last_frame_->time_stamp_);
        imu_preintegrator_from_RefKF_->update(imu[0].gyro_ - bg, imu[0].acce_ - ba, dt);
    }

    for (size_t i = 0; i < imu.size(); i++) {
        double nextt;
        if (i == imu.size() - 1)
            nextt = current_frame_->time_stamp_;
        else
            nextt = imu[i+1].timestamp_;  // regular condition, next is imu data
        // delta time
        double dt = std::max(0., nextt - imu[i].timestamp_);
        // update pre-integrator      
        imu_preintegrator_from_RefKF_->update(imu[i].gyro_ - bg, imu[i].acce_ - ba, dt);
    }

    last_keyframe_->is_bias_updated_recently_ = false;
}

bool Frontend::IMUInitialization() {
    // Step0. get all keyframes in map
    //        reset v/bg/ba to 0
    //        re-compute pre-integration
    auto vpKFs = map_->GetAllKeyFrames();
    int N = vpKFs.size();
    if (N < num_kfs_for_imu_init_) return false; 

    // Step1. gyroscope bias estimation
    //        update bg and re-compute pre-integration
    Vector3d bgest = IMUInitEstBg(vpKFs);

    for (auto& [_, KF]: vpKFs) {
        KF->SetBiasG(bgest);
    }
    for (int i = 1; i < N; i++) {
        vpKFs[i]->ReComputeIMUPreIntegration();
    }

    // Step2. accelerometer bias and gravity estimation (gv = Rvw*gw)
    // let's first assume ba is given by prior and solve the gw
    // Step 2.1 gravity estimation

    // Solve C*x=D for x=[gw] (3+3)x1 vector
    // \see section IV in "Visual Inertial Monocular SLAM with Map Reuse"
    Vector3d baPrior = settings::biasAccePrior;

    MatrixXd C(3 * (N - 2), 3);
    C.setZero();

    VectorXd D(3 * (N - 2));
    D.setZero();

    Matrix3d I3 = Matrix3d::Identity();
    for (int i = 0; i < N - 2; i++) {

        // 三个帧才能建立加速度约束
        std::shared_ptr<Frame> pKF1 = vpKFs[i];
        std::shared_ptr<Frame> pKF2 = vpKFs[i + 1];
        std::shared_ptr<Frame> pKF3 = vpKFs[i + 2];

        // Poses
        Matrix3d R1 = pKF1->ImuPose().rotationMatrix().matrix();
        Matrix3d R2 = pKF2->ImuPose().rotationMatrix().matrix();
        Vector3d p1 = pKF1->ImuPose().translation().matrix();
        Vector3d p2 = pKF2->ImuPose().translation().matrix();
        Vector3d p3 = pKF3->ImuPose().translation().matrix();

        // Delta time between frames
        double dt12 = pKF2->GetIMUPreInt()->getDeltaTime();
        double dt23 = pKF3->GetIMUPreInt()->getDeltaTime();
        // Pre-integrated measurements
        Vector3d dp12 = pKF2->GetIMUPreInt()->getDeltaP();
        Vector3d dv12 = pKF2->GetIMUPreInt()->getDeltaV();
        Vector3d dp23 = pKF3->GetIMUPreInt()->getDeltaP();

        Matrix3d Jpba12 = pKF2->GetIMUPreInt()->getJPBiasa();
        Matrix3d Jvba12 = pKF2->GetIMUPreInt()->getJVBiasa();
        Matrix3d Jpba23 = pKF3->GetIMUPreInt()->getJPBiasa();

        // 谜之计算
        Matrix3d lambda = 0.5 * (dt12 * dt12 * dt23 + dt12 * dt23 * dt23) * I3;
        Vector3d phi = R2 * Jpba23 * baPrior * dt12 -
                       R1 * Jpba12 * baPrior * dt23 +
                       R1 * Jvba12 * baPrior * dt12 * dt23;
        Vector3d gamma = p3 * dt12 + p1 * dt23 + R1 * dp12 * dt23 - p2 * (dt12 + dt23)
                         - R2 * dp23 * dt12 - R1 * dv12 * dt12 * dt23;

        C.block<3, 3>(3 * i, 0) = lambda;
        D.segment<3>(3 * i) = gamma - phi;
    }

    // Use svd to compute C*x=D, x=[ba] 6x1 vector
    // Solve Ax = b where x is ba
    Eigen::JacobiSVD<MatrixXd> svd2(C, Eigen::ComputeThinU | Eigen::ComputeThinV);
    VectorXd y = svd2.solve(D);
    Vector3d gpre = y.head(3);
    // normalize g
    Vector3d g0 = gpre / gpre.norm() * settings::GRAVITY_VALUE;
    
    // Step2.2
    // estimate the bias from g
    MatrixXd A(3 * (N - 2), 3);
    A.setZero();
    VectorXd B(3 * (N - 2));
    B.setZero();

    for (int i = 0; i < N - 2; i++) {

        // 三个帧才能建立加速度约束
        std::shared_ptr<Frame> pKF1 = vpKFs[i];
        std::shared_ptr<Frame> pKF2 = vpKFs[i + 1];
        std::shared_ptr<Frame> pKF3 = vpKFs[i + 2];

        // Poses
        Matrix3d R1 = pKF1->ImuPose().rotationMatrix().matrix();
        Matrix3d R2 = pKF2->ImuPose().rotationMatrix().matrix();
        Vector3d p1 = pKF1->ImuPose().translation().matrix();
        Vector3d p2 = pKF2->ImuPose().translation().matrix();
        Vector3d p3 = pKF3->ImuPose().translation().matrix();

        // Delta time between frames
        double dt12 = pKF2->GetIMUPreInt()->getDeltaTime();
        double dt23 = pKF3->GetIMUPreInt()->getDeltaTime();
        // Pre-integrated measurements
        Vector3d dp12 = pKF2->GetIMUPreInt()->getDeltaP();
        Vector3d dv12 = pKF2->GetIMUPreInt()->getDeltaV();
        Vector3d dp23 = pKF3->GetIMUPreInt()->getDeltaP();

        Matrix3d Jpba12 = pKF2->GetIMUPreInt()->getJPBiasa();
        Matrix3d Jvba12 = pKF2->GetIMUPreInt()->getJVBiasa();
        Matrix3d Jpba23 = pKF3->GetIMUPreInt()->getJPBiasa();

        // 谜之计算
        Vector3d lambda = 0.5 * (dt12 * dt12 * dt23 + dt12 * dt23 * dt23) * I3 * g0;
        Matrix3d phi = R2 * Jpba23 * dt12 -
                       R1 * Jpba12 * dt23 +
                       R1 * Jvba12 * dt12 * dt23;
        Vector3d gamma = p3 * dt12 + p1 * dt23 + R1 * dp12 * dt23 - p2 * (dt12 + dt23)
                         - R2 * dp23 * dt12 - R1 * dv12 * dt12 * dt23;

        A.block<3, 3>(3 * i, 0) = phi;
        B.segment<3>(3 * i) = gamma - lambda;
    }

    Eigen::JacobiSVD<MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    VectorXd y2 = svd.solve(B);
    Vector3d baest = y2;

    // update ba and re-compute pre-integration
    for (auto& [_, pkf]: vpKFs) {
        pkf->SetBiasA(baest);
    }
    for (int i = 1; i < N; i++) {
        vpKFs[i]->ReComputeIMUPreIntegration();
    }

    // Step3. velocity estimation
    for (int i = 0; i < N; i++) {
        auto pKF = vpKFs[i];
        if (i != N - 1) {
            // not last KeyFrame, R1*dp12 = p2 - p1 -v1*dt12 - 0.5*gw*dt12*dt12
            //  ==>> v1 = 1/dt12 * (p2 - p1 - 0.5*gw*dt12*dt12 - R1*dp12)

            auto pKF2 = vpKFs[i + 1];
            const Vector3d p2 = pKF2->ImuPose().translation().matrix();
            const Vector3d p1 = pKF->ImuPose().translation().matrix();
            const Matrix3d R1 = pKF->ImuPose().rotationMatrix().matrix();
            const double dt12 = pKF2->GetIMUPreInt()->getDeltaTime();
            const Vector3d dp12 = pKF2->GetIMUPreInt()->getDeltaP();

            Vector3d v1 = (p2 - p1 - 0.5 * g0 * dt12 * dt12 - R1 * dp12) / dt12;
            pKF->SetVelocity(v1);
        } else {
            // last KeyFrame, R0*dv01 = v1 - v0 - gw*dt01 ==>> v1 = v0 + gw*dt01 + R0*dv01
            auto pKF0 = vpKFs[i - 1];
            const Matrix3d R0 = pKF0->ImuPose().rotationMatrix().matrix();
            const Vector3d v0 = pKF0->Velocity();
            const double dt01 = pKF->GetIMUPreInt()->getDeltaTime();
            const Vector3d dv01 = pKF->GetIMUPreInt()->getDeltaV();

            Vector3d v1 = v0 + g0 * dt01 + R0 * dv01;
            pKF->SetVelocity(v1);
        }
    }

    double gprenorm = gpre.norm();
    // double baestdif = (baest0 - baest).norm();

    bool initflag = false;
    if (gprenorm > 9.7 && gprenorm < 9.9 && /* baestdif < 0.2  && */
        baest.norm() < 1) {
        initflag = true;
    } else {
        // goodcnt = 0;
    }

    // align 'world frame' to gravity vector, making mgWorld = [0,0,9.8]
    if (initflag) {
        /*
        // compute Rvw
        Vector3d gw1(0, 0, 1);
        Vector3d gv1 = g0 / g0.norm();
        Vector3d gw1xgv1 = gw1.cross(gv1);
        Vector3d vhat = gw1xgv1 / gw1xgv1.norm();
        double theta = std::atan2(gw1xgv1.norm(), gw1.dot(gv1));
        Matrix3d Rvw = Sophus::SO3d::exp(vhat * theta).matrix();
        Matrix3d Rwv = Rvw.transpose();
        Sophus::SE3d Twv(Rwv, Vector3d::Zero());
        // 设置重力
        Vector3d gw = Rwv * g0;
        mgWorld = gw;

        // rotate pose/rotation/velocity to align with 'world' frame
        for (int i = 0; i < N; i++) {
            auto pKF = vpKFs[i];
            Sophus::SE3d Tvb = pKF->GetPose();
            Vector3d Vvb = pKF->Speed();
            // set pose/speed/biasg/biasa
            pKF->SetPose(Twv * Tvb);
            pKF->SetVelocity(Rwv * Vvb);
            pKF->SetBiasG(bgest);
            pKF->SetBiasA(baest);
        }

        if (mpCurrentFrame->IsKeyFrame() == false) {
            mpCurrentFrame->SetPose(Twv * mpCurrentFrame->GetPose());
            mpCurrentFrame->SetVelocity(Rwv * mpCurrentFrame->Speed());
            mpCurrentFrame->SetBiasG(bgest);
            mpCurrentFrame->SetBiasA(baest);
        }

        // re-compute pre-integration for KeyFrame (except for the first KeyFrame)
        for (int i = 1; i < N; i++) {
            vpKFs[i]->ComputeIMUPreInt();
        }

        // MapPoints
        auto vsMPs = mpBackEnd->GetLocalMap();
        for (auto mp : vsMPs) {
            Vector3d Pv = mp->GetWorldPos();
            Vector3d Pw = Rwv * Pv;
            mp->SetWorldPos(Pw);
        }
         */
        g_= g0;
        map_->is_imu_initialized_ = true;
        backend_->SetGravity(g0);
        LOG(INFO) << "Frontend::IMUInitialization(): Successfully initialized IMU!";
        LOG(INFO) << "Estimated gravity before: " << gpre.transpose() << ", |gw| = " << gprenorm ;
        LOG(INFO) << "Estimated acc bias after: " << baest.transpose() ;
        LOG(INFO) << "Estimated gyr bias: " << bgest.transpose();
    }
    return initflag;
}

Vector3d Frontend::IMUInitEstBg(const std::unordered_map<u_long, std::shared_ptr<Frame>>& vpKFs) {
    typedef g2o::BlockSolverX BlockSolverType;
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType> (g2o::make_unique<LinearSolverType>())
    );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    
    // Add vertex of gyro bias, to optimizer graph
    VertexGyroBias *vertex_biasg = new VertexGyroBias();
    vertex_biasg->setId(0);
    vertex_biasg->setEstimate(Eigen::Vector3d::Zero());
    optimizer.addVertex(vertex_biasg);

    // Add unary edges for gyro bias vertex
    assert(vpKFs.size() > 0);
    for (auto& [_ , pKF] : vpKFs) {
        // skip the first KF
        if (_ == 0) continue;

        assert(pKF->reference_KF_.expired() == false);
        std::shared_ptr<Frame> pPrevKF = pKF->reference_KF_.lock();

        auto& imupreint = pKF->GetIMUPreInt();
        assert(imupreint != nullptr && imupreint->getDeltaTime() > 0);
        EdgeGyroBias *edge_biasg = new EdgeGyroBias();
        edge_biasg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));

        edge_biasg->dRbij = imupreint->getDeltaR();
        edge_biasg->J_dR_bg = imupreint->getJRBiasg();
        edge_biasg->Rwbi = pPrevKF->ImuPose().rotationMatrix();
        edge_biasg->Rwbj = pKF->ImuPose().rotationMatrix();
        edge_biasg->setInformation(imupreint->getCovPVPhi().bottomRightCorner(3, 3).inverse());
        optimizer.addEdge(edge_biasg);
    }

    // It's actualy a linear estimator, so 1 iteration is enough.
    optimizer.initializeOptimization();
    optimizer.optimize(1);

    // update bias G
    VertexGyroBias *vBgEst = static_cast<VertexGyroBias *>(optimizer.vertex(0));

    return vBgEst->estimate();
}


} // namespace demoam 