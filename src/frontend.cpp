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
    
    imu_preintegrator_from_RefKF_ = std::make_shared<IMUPreintegration>();
    imu_preintegrator_from_lastframe_ = std::make_shared<IMUPreintegration>();

    num_features_init_ = Config::Get<int>("num_features_init");
    num_features_tracking_good_ = Config::Get<int>("num_features_tracking_good");
    num_features_needed_for_keyframe_ = Config::Get<int>("num_features_needed_for_keyframe");

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

    if (status_ != FrontendStatus::RECENTLY_LOST && status_ != FrontendStatus::LOST) {
        last_frame_ = current_frame_;
    }

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
        if (TriangulatePoints(poses, points, pw)) { // https://www.cvlibs.net/datasets/kitti/setup.php //TODO:ProblemSolved
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

    if (status_ != FrontendStatus::RECENTLY_LOST) {
        SearchLastFrameOpticalFlow();
        tracking_inliers_ = EstimateCurrentPose();
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

    if (map_->isImuInitialized() == false) 
        IMUInitialization();

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

int Frontend::EstimateCurrentPose() {
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
    LOG(INFO) << "Frontend::EstimateCurrentPose(): Outlier/Inlier in pose estimating: " << cnt_outlier << "/" << features.size() - cnt_outlier;

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
/**
 * //TODO: KF insertion
 * @brief 判断当前帧是否需要插入关键帧
 * 
 * Step 1：纯VO模式下不插入关键帧，如果局部地图被闭环检测使用，则不插入关键帧
 * Step 2：如果距离上一次重定位比较近，或者关键帧数目超出最大限制，不插入关键帧
 * Step 3：得到参考关键帧跟踪到的地图点数量
 * Step 4：查询局部地图管理器是否繁忙,也就是当前能否接受新的关键帧
 * Step 5：对于双目或RGBD摄像头，统计可以添加的有效地图点总数 和 跟踪到的地图点数量
 * Step 6：决策是否需要插入关键帧
 * @return true         需要
 * @return false        不需要
 */
    if (tracking_inliers_ >= num_features_needed_for_keyframe_) {
        return false;
    }
    
    current_frame_ -> SetKeyFrame();
    map_ -> InsertKeyFrame(current_frame_);

    LOG(INFO) << "Frontend::InsertKeyFrame(): Set frame " << current_frame_ -> id_ << " as keyframe " << current_frame_ -> keyframe_id_;

    SetObservationsForKeyFrame();
    DetectFastInLeft();
    SearchInRightOpticalFlow();
    TriangulateNewPoints();

    backend_-> UpdateMap();

    

    if (viewer_) viewer_ -> UpdateMap();

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

    for (auto mp : localmap)
        if (mp)
            mp->mbTrackInView = false;
    for (auto feat: mpCurrentFrame->mFeaturesLeft)
        if (feat && feat->mpPoint && feat->mbOutlier == false)
            feat->mpPoint->mbTrackInView = true;
    // 筛一下视野内的点
    set<shared_ptr<MapPoint> > mpsInView;
    for (auto &mp: localmap) {
        if (mp && mp->isBad() == false && mp->mbTrackInView == false && mpCurrentFrame->isInFrustum(mp, 0.5)) {
            mpsInView.insert(mp);
        }
    }
    if (mpsInView.empty())
        return tracking_inliers_ >= setting::minTrackLocalMapInliers;
    LOG(INFO) << "Call Search by direct projection" << endl;
    int cntMatches = mpMatcher->SearchByDirectProjection(mpCurrentFrame, mpsInView);
    LOG(INFO) << "Track local map matches: " << cntMatches << ", current features: "
              << mpCurrentFrame->mFeaturesLeft.size() << endl;
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    timeCost = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    LOG(INFO) << "Search local map points cost time: " << timeCost << endl;
    // Optimize Pose
    int optinliers = OptimizeCurrentPose();
    // Update MapPoints Statistics
    tracking_inliers_ = 0;
    for (shared_ptr<Feature> feat : mpCurrentFrame->mFeaturesLeft) {
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
    //TODO: PredictCurrentPose
    if (map_->isImuInitialized() == false) {
        current_frame_ -> SetPose(relative_motion_ * last_frame_ -> Pose());
        return;
    }
    
    current_frame_->SetPose(last_keyframe_->Pose());
    current_frame_->SetVelocitynBias(last_keyframe_->Velocity(), last_keyframe_->BiasG(), last_keyframe_->BiasA());

   // current_frame_->IMUPreintegration

}

void Frontend::PreintegrateIMU() { 
    if (current_frame_->imu_meas_.empty()) {
        return;
        // like the first frame 
    }
    imu_meas_since_RefKF_.insert(imu_meas_since_RefKF_.end(), 
                                   current_frame_->imu_meas_.begin(), 
                                   current_frame_->imu_meas_.end());

    imu_preintegrator_from_lastframe_.reset(new IMUPreintegration(current_frame_->BiasG(), current_frame_->BiasA()));

    const auto &imu = current_frame_->imu_meas_;

    {   // consider the gap between the last KF and the first IMU
        // delta time
        double dt = std::max(0., imu[0].timestamp_ - last_frame_->time_stamp_);
        // update pre-integrator
        imu_preintegrator_from_lastframe_->Integrate(imu[0], dt);
        imu_preintegrator_from_RefKF_->Integrate(imu[0], dt);
        
    }

    for (size_t i = 0; i < current_frame_->imu_meas_.size(); i++) {
        double nextt;
        if (i == imu.size() - 1)
            nextt = current_frame_->time_stamp_;
        else
            nextt = imu[i+1].timestamp_;  // regular condition, next is imu data
        // delta time
        double dt = std::max(0., nextt - imu[i].timestamp_);
        // update pre-integrator
        imu_preintegrator_from_lastframe_->Integrate(imu[i], dt);
        imu_preintegrator_from_RefKF_->Integrate(imu[i], dt);
    }
    
}

void Frontend::IMUInitialization() {
    return;
    //TODO: IMUInitialization

    // Step0. get all keyframes in map
    //        reset v/bg/ba to 0
    //        re-compute pre-integration
    auto all_KFs = map_->GetAllKeyFrames();
    int n = all_KFs.size();
    if (n < settings::NUM_KFs_FOR_IMU_INIT) return; 

    // Step1. gyroscope bias estimation
    //        update bg and re-compute pre-integration
    Vector3d bgest = IMUInitEstBg(all_KFs);
    for (auto& [_, KF]: all_KFs) {
        KF->SetBiasG(bgest);
    }
    for (int i = 1; i < n; i++) {
        all_KFs[i]->ReComputeIMUPreIntegration();
    }

    // Step2. accelerometer bias and gravity estimation (gv = Rvw*gw)
    // let's first assume ba is given by prior and solve the gw
    // Step 2.1 gravity estimation

    // Solve C*x=D for x=[gw] (3+3)x1 vector
    // \see section IV in "Visual Inertial Monocular SLAM with Map Reuse"

}

Vector3d Frontend::IMUInitEstBg(const std::map<u_long, std::shared_ptr<Frame>>& vpKFs) {
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
    std::shared_ptr<Frame> pPrevKF0 = vpKFs.begin()->second;
    for (auto& [_ , pKF] : vpKFs) {
        // Ignore the first KF
        if (_ == vpKFs.begin()->first)
            continue;

        std::shared_ptr<Frame> pPrevKF = pKF->reference_KF_.lock();

        auto& imupreint = pKF->IMUPreintegrator();
        EdgeGyroBias *edge_biasg = new EdgeGyroBias();
        edge_biasg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));

        // measurement is not used in EdgeGyrBias
        edge_biasg->dRbij = imupreint->GetDeltaRotation(imupreint->bg_).matrix();
        edge_biasg->J_dR_bg = imupreint->dR_dbg_;
        edge_biasg->Rwbi = pPrevKF->Pose().rotationMatrix();
        edge_biasg->Rwbj = pKF->Pose().rotationMatrix();
        edge_biasg->setInformation(imupreint->cov_.bottomRightCorner(3, 3).inverse());
        optimizer.addEdge(edge_biasg);

        pPrevKF0 = pKF;
    }

    // It's actualy a linear estimator, so 1 iteration is enough.
    optimizer.initializeOptimization();
    optimizer.optimize(1);

    // update bias G
    VertexGyroBias *vBgEst = static_cast<VertexGyroBias *>(optimizer.vertex(0));

    return vBgEst->estimate();
}


} // namespace demoam 