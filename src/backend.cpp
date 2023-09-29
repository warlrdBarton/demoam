#include "backend.h"

#include <glog/logging.h>

#include "geometric_tools.h"
#include "feature.h"
#include "g2o_types.h"
#include "map.h"
#include "mappoint.h"
#include "frame.h"
#include "camera.h"

namespace demoam {
    
Backend::Backend() {
    backend_running_.store(true);
    backend_thread_ = std::thread(&Backend::BackendLoop, this);
}

void Backend::UpdateMap() {
    std::unique_lock<std::mutex> lck(data_mutex_);
    map_update_.notify_one();
}

void Backend::Stop() {
    backend_running_.store(false);
    map_update_.notify_one();
    backend_thread_.join();
}

bool Backend::IsCurrentlyBusy() {
    std::unique_lock<std::mutex> lck(data_mutex_);
    return is_busy_;
}

void Backend::BackendLoop() {
    while (backend_running_.load()) {
        std::unique_lock<std::mutex> lck(data_mutex_);
        map_update_.wait(lck);
        is_busy_ = true;
        std::unordered_map<u_long, std::shared_ptr<Frame>> active_keyframes = map_ -> GetActiveKeyFrames();
        std::unordered_map<u_long, std::shared_ptr<MapPoint>> active_mappoints = map_ -> GetActiveMapPoints();
        Optimize(active_keyframes, active_mappoints);
        is_busy_ = false;
    }
}

void Backend::Optimize(std::unordered_map<u_long, std::shared_ptr<Frame>>& keyframes, std::unordered_map<u_long, std::shared_ptr<MapPoint>>& mappoints) {
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    u_long max_kf_id = 0;
    std::unordered_map<u_long, VertexSE3Expmap*> vertices;
    for (auto& [_, kf] : keyframes) {
        VertexSE3Expmap* vertex_pose = new VertexSE3Expmap();
        vertex_pose -> setId(kf -> keyframe_id_);
        vertex_pose -> setEstimate(kf -> Pose());
        optimizer.addVertex(vertex_pose);
        vertices.insert({kf -> keyframe_id_, vertex_pose});
        max_kf_id = std::max(max_kf_id, kf -> keyframe_id_);
    }
    
    Eigen::Matrix3d K = camera_left_ -> K();
    Sophus::SE3d left_extrinsics = camera_left_ -> Pose();
    Sophus::SE3d right_extrinsics = camera_right_ -> Pose();


    std::unordered_map<u_long, VertexSBAPointXYZ*> vertices_mappoints;
    std::vector<EdgeSE3ProjectXYZ*> edges;
    std::vector<std::shared_ptr<Feature>> features;
    int index = 0; 
    
    for (auto& [_, mp] : mappoints) {
        if (mp -> is_outlier_) {
            LOG(WARNING) << "MAPPOINT OUTLIER ENCOUNTER";
            continue;
        }

        if (vertices_mappoints.count(mp -> id_) == false) {
            VertexSBAPointXYZ* v = new VertexSBAPointXYZ;
            v -> setEstimate(mp -> Pos());
            v -> setId(mp -> id_ + max_kf_id + 1);
            v -> setMarginalized(true);
            optimizer.addVertex(v);
            vertices_mappoints.insert({mp -> id_, v});
        }

        for (auto& obs : mp -> Observations()) {
            auto feat = obs.lock();
            if (!feat || feat -> is_outlier_ || !feat -> frame_.lock()) continue;
            auto frame = feat -> frame_.lock();
            EdgeSE3ProjectXYZ* edge = new EdgeSE3ProjectXYZ(K, feat -> is_on_left_image_ ? left_extrinsics : right_extrinsics);
            edge -> setId(index);
            edge -> setVertex(0, vertices[frame -> keyframe_id_]);
            edge -> setVertex(1, vertices_mappoints[mp -> id_]);
            edge -> setMeasurement(Eigen::Vector2d(feat -> position_.pt.x, feat -> position_.pt.y));
            edge -> setInformation(Eigen::Matrix2d::Identity());
            edge -> setRobustKernel(new g2o::RobustKernelHuber);
            optimizer.addEdge(edge);
            edges.push_back(edge);
            features.push_back(feat);
            index++;
        }
    }
     
    const double chi2_th = 5.991; // chi2 val for Prob 0.05 with 2-Dof;
    int cnt_outlier = 0;

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    for (size_t i = 0; i < edges.size(); ++i) {
        if (edges[i] -> chi2() > chi2_th) {
            features[i] -> is_outlier_ = true;
            cnt_outlier++;
            features[i] -> mappoint_.lock() -> RemoveObservation(features[i]);
            features[i] -> mappoint_.reset();
        }
    }

    LOG(INFO) << "Backend::Optimize(): Outlier/Inlier in optimization: " << cnt_outlier << "/" << features.size() - cnt_outlier;

    for (auto& [kf_id, v] : vertices) {
        keyframes[kf_id] -> SetPose(v -> estimate());
    }
    for (auto& [mp_id, v] : vertices_mappoints) {
        mappoints[mp_id] -> SetPos(v -> estimate());
    }
    
}

} // namespace demoam