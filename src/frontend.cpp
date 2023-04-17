#include "frontend.h"
#include "algorithm.h"
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
#include "feature_tracker.h"


namespace demoam {
Frontend::Frontend() {
    num_features_init_ = Config::Get<int>("num_features_init");
    num_features_ = Config::Get<int>("num_features");
    save_to_file_.open("./traj.txt", std::ios::trunc);
}

void Frontend::Stop() {
    save_to_file_.close();
}

bool Frontend::AddFrame(std::shared_ptr<Frame> frame) {
    current_frame_ = frame;
    switch (status_) {
        case FrontendStatus::INITING:
            StereoInit();
            break;
        case FrontendStatus::TRACKING_GOOD:
        case FrontendStatus::TRACKING_BAD:
            Track();
            break;
        case FrontendStatus::LOST:
            Reset();
            break;
    }
    SaveTrajectoryKITTI();
    last_frame_ = current_frame_;
    return true;
}

bool Frontend::StereoInit() {
    FeatureTracker::DetectFastInLeft(current_frame_);
    if (FeatureTracker::SearchInRightOpticalFlow(current_frame_, camera_right_) < num_features_init_) {
        return false;
    } 
    bool build_map_success = BuildInitMap();
    if (build_map_success) {
        status_ = FrontendStatus::TRACKING_GOOD;
        if (viewer_) {
            viewer_ -> AddCurrentFrame(current_frame_);
            viewer_ -> UpdateMap();
        }
        return true;
    }
    return false;       
}

bool Frontend::BuildInitMap() {
    int cnt_init_mappoints = 0;
    std::vector<Sophus::SE3d> poses{camera_left_ -> Pose(), camera_right_ -> Pose()}; 
    for (size_t i = 0; i < current_frame_ -> features_left_.size(); ++i) {
        if (current_frame_ -> features_right_[i] == nullptr) continue;
        std::vector<Eigen::Vector3d> points{
            camera_left_ -> pixel2camera(
                Eigen::Vector2d(current_frame_->features_left_[i] -> position_.pt.x,
                                current_frame_->features_left_[i] -> position_.pt.y)),
            camera_right_ -> pixel2camera(
                Eigen::Vector2d(current_frame_->features_right_[i] -> position_.pt.x,
                                current_frame_->features_right_[i] -> position_.pt.y))
        };
        Eigen::Vector3d pw = Eigen::Vector3d::Zero();
        if (TriangulatePoints(poses, points, pw) && pw[2] > 0) {
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
    if (last_frame_) {
        current_frame_ -> SetPose(relative_motion_ * last_frame_ -> Pose());
    }
    FeatureTracker::SearchLastFrameOpticalFlow(last_frame_, current_frame_, camera_left_);
    tracking_inliners_ = EstimateCurrentPose();
    
    if (tracking_inliners_ > num_features_tracking_) {
        status_  = FrontendStatus::TRACKING_GOOD;
    } else if (tracking_inliners_ > num_features_tracking_bad_) {
        status_ = FrontendStatus::TRACKING_BAD;
    } else {
        status_ = FrontendStatus::LOST;
    }
    
    InsertKeyFrame();
    relative_motion_ = current_frame_ -> Pose() * last_frame_ -> Pose().inverse();
    
    if (viewer_) viewer_ -> AddCurrentFrame(current_frame_);
    return true; 
}

int Frontend::EstimateCurrentPose() {
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    VertexPose *vertex_pose = new VertexPose();
    vertex_pose -> setId(0);
    vertex_pose -> setEstimate(current_frame_ -> Pose());
    optimizer.addVertex(vertex_pose);

    Eigen::Matrix3d K = camera_left_ -> K();

    int index = 0; 
    std::vector<EdgeProjectionPoseOnly *> edges;
    std::vector<std::shared_ptr<Feature>> features;
    for (auto& feat : (current_frame_ -> features_left_)) {
        auto mp = feat -> mappoint_.lock();
        if (mp) {
            features.push_back(feat);
            EdgeProjectionPoseOnly* edge = new EdgeProjectionPoseOnly(mp -> pos_, K);
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
    if (tracking_inliners_ >= num_features_needed_for_keyframe_) {
        return false;
    }
    current_frame_ -> SetKeyFrame();
    map_ -> InsertKeyFrame(current_frame_);

    LOG(INFO) << "Frontend::InsertKeyFrame(): Set frame " << current_frame_ -> id_ << " as keyframe " << current_frame_ -> keyframe_id_;

    SetObservationsForKeyFrame();
    FeatureTracker::DetectFastInLeft(current_frame_);
    FeatureTracker::SearchInRightOpticalFlow(current_frame_, camera_right_);
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
    std::vector<Sophus::SE3d> poses{camera_left_ -> Pose(), camera_right_ -> Pose()}; 
    for (size_t i = 0; i < current_frame_ -> features_left_.size(); ++i) {
        if (current_frame_ -> features_left_[i] -> mappoint_.expired() == false
            || current_frame_ -> features_right_[i] == nullptr) {
                continue;
        }
        std::vector<Eigen::Vector3d> points{
            camera_left_ -> pixel2camera(
                Eigen::Vector2d(current_frame_->features_left_[i] -> position_.pt.x,
                                current_frame_->features_left_[i] -> position_.pt.y)),
            camera_right_ -> pixel2camera(
                Eigen::Vector2d(current_frame_->features_right_[i] -> position_.pt.x,
                                current_frame_->features_right_[i] -> position_.pt.y))
        };
        Eigen::Vector3d pw = Eigen::Vector3d::Zero();
        if (TriangulatePoints(poses, points, pw) && pw[2] > 0) {
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
    save_to_file_ << std::endl;
}

}