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
    if (map_->isImuInitialized()) {
        LocalInertialBA(keyframes, mappoints);
    } else {
        LocalBundleAdjustment(keyframes, mappoints);
    }
}

void Backend::LocalBundleAdjustment(std::unordered_map<u_long, std::shared_ptr<Frame>>& keyframes, std::unordered_map<u_long, std::shared_ptr<MapPoint>>& mappoints) {
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

void Backend::LocalInertialBA(std::unordered_map<u_long, std::shared_ptr<Frame>>& keyframes, std::unordered_map<u_long, std::shared_ptr<MapPoint>>& mappoints) {
    // Bundle adjustment with IMU
    LOG(INFO) << "Call local ba XYZ with imu";
    // Gravity vector in world frame
    Vector3d GravityVec = g_;
    // Setup optimizer
    typedef g2o::BlockSolverX BlockSolverType;
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    int maxKFid = 0;
    for (auto& [_, pKFi] : keyframes) {
        int idKF = pKFi->keyframe_id_ * 4;
        if (idKF + 3 > maxKFid) {
            maxKFid = idKF + 3;
        }
        // P+R
        VertexPose *vPR = new VertexPose();
        vPR->setEstimate(pKFi->ImuPose());
        vPR->setId(idKF);
        optimizer.addVertex(vPR);
        // speed
        VertexVelocity *vSpeed = new VertexVelocity();
        vSpeed->setEstimate(pKFi->Velocity());
        vSpeed->setId(idKF + 1);
        optimizer.addVertex(vSpeed);
        // Bg
        VertexGyroBias *vBg = new VertexGyroBias();
        vBg->setId(idKF + 2);
        vBg->setEstimate(pKFi->BiasG());
        optimizer.addVertex(vBg);
        // Ba
        VertexAccBias *vBa = new VertexAccBias();
        vBa->setId(idKF + 3);
        vBa->setEstimate(pKFi->BiasA());
        optimizer.addVertex(vBa);

        // fix the first one
        if (_ == 0) {
            vPR->setFixed(true);
        } 
    }

    // 关键帧之间的边
    vector<EdgePRV *> vpEdgePRV;
    vector<EdgeBiasG *> vpEdgeBg;
    vector<EdgeBiasA *> vpEdgeBa;

    // Use chi2inv() in MATLAB to compute the value corresponding to 0.95/0.99 prob. w.r.t 15DOF: 24.9958/30.5779
    // 12.592/16.812 for 0.95/0.99 6DoF
    // 16.919/21.666 for 0.95/0.99 9DoF
    const float thHuberPRV = sqrt(1500 * 21.666);
    const float thHuberBias = sqrt(1500 * 16.812);
    // Inverse covariance of bias random walk
    Matrix3d infoBg = Matrix3d::Identity() / settings::gyrBiasRw2;
    Matrix3d infoBa = Matrix3d::Identity() / settings::accBiasRw2;
    for (auto& [_, pKF1] : keyframes) {
        if (pKF1->reference_KF_.expired()) {
            if (_ != 0) {
                LOG(ERROR) << "non-first KeyFrame has no reference KF";
            }
            continue;
        }
        std::shared_ptr<Frame> pKF0 = pKF1->reference_KF_.lock();   // Previous KF
        auto& M = *(pKF1->GetIMUPreInt());

        // PR0, PR1, V0, V1, Bg0, Ba0
        EdgePRV *ePRV = new EdgePRV(GravityVec);
        ePRV->setVertex(0, optimizer.vertex(pKF0->keyframe_id_ * 4));
        ePRV->setVertex(1, optimizer.vertex(pKF1->keyframe_id_ * 4));
        ePRV->setVertex(2, optimizer.vertex(pKF0->keyframe_id_ * 4 + 1));
        ePRV->setVertex(3, optimizer.vertex(pKF1->keyframe_id_ * 4 + 1));
        ePRV->setVertex(4, optimizer.vertex(pKF0->keyframe_id_ * 4 + 2));
        ePRV->setVertex(5, optimizer.vertex(pKF0->keyframe_id_ * 4 + 3));
        ePRV->setMeasurement(M);
        // set Covariance
        Matrix9d CovPRV = M.getCovPVPhi();
        // 但是Edge里用是P,R,V，所以交换顺序
        CovPRV.col(3).swap(CovPRV.col(6));
        CovPRV.col(4).swap(CovPRV.col(7));
        CovPRV.col(5).swap(CovPRV.col(8));
        CovPRV.row(3).swap(CovPRV.row(6));
        CovPRV.row(4).swap(CovPRV.row(7));
        CovPRV.row(5).swap(CovPRV.row(8));
        ePRV->setInformation(CovPRV.inverse());
        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
        ePRV->setRobustKernel(rk);
        rk->setDelta(thHuberPRV);
        optimizer.addEdge(ePRV);
        vpEdgePRV.push_back(ePRV);

        // bias 的随机游走，用两条边来约束
        double dt = M.getDeltaTime();

        EdgeBiasG *eBG = new EdgeBiasG();
        eBG->setVertex(0, optimizer.vertex(pKF0->keyframe_id_ * 4 + 2));
        eBG->setVertex(1, optimizer.vertex(pKF1->keyframe_id_ * 4 + 2));
        eBG->setMeasurement(Vector3d::Zero());   
        eBG->setInformation(infoBg / dt);
        g2o::RobustKernelHuber *rkb = new g2o::RobustKernelHuber;
        eBG->setRobustKernel(rkb);
        rkb->setDelta(thHuberBias);
        optimizer.addEdge(eBG);
        vpEdgeBg.push_back(eBG);

        EdgeBiasA *eBA = new EdgeBiasA();
        eBA->setVertex(0, optimizer.vertex(pKF0->keyframe_id_ * 4 + 3));
        eBA->setVertex(1, optimizer.vertex(pKF1->keyframe_id_ * 4 + 3));
        eBA->setMeasurement(Vector3d::Zero());   
        eBA->setInformation(infoBa / dt);
        g2o::RobustKernelHuber *rkba = new g2o::RobustKernelHuber;
        eBA->setRobustKernel(rkba);
        rkba->setDelta(thHuberBias);
        optimizer.addEdge(eBA);
        vpEdgeBa.push_back(eBA);
    }

    // Set MapPoint vertices
    vector<EdgePRXYZ *> vpEdgePoints;
    vpEdgePoints.reserve(mappoints.size());
    vector<std::shared_ptr<MapPoint> > vpMappoints;
    vector<std::shared_ptr<Feature>> vpFeatures;
    vpMappoints.reserve(mappoints.size());
    vpFeatures.reserve(mappoints.size());
    const float thHuber = sqrt(5.991);  // 0.95 chi2
    const float thHuber2 = 5.991;  // 0.95 chi2
    for (auto& [_, mp] : mappoints) {
        if (mp == nullptr)
            continue;
        if (mp->is_outlier_)
            continue;
        if (mp->observed_times_ > 1) {
            VertexSBAPointXYZ *vXYZ = new VertexSBAPointXYZ;
            int idMP = mp->id_ + maxKFid + 1;
            vXYZ->setId(idMP);
            vXYZ->setEstimate(mp->Pos());
            vXYZ->setMarginalized(true);
            optimizer.addVertex(vXYZ);
            int index = 0;
            // add edges in observation
            for (auto& obs : mp -> Observations()) {
                if (obs.expired())
                    continue;
                auto feat = obs.lock();
                if (!feat || feat -> is_outlier_ || !feat -> frame_.lock()) continue;
                auto kf = feat -> frame_.lock();
                
                EdgePRXYZ *eProj = new EdgePRXYZ(feat -> is_on_left_image_ ? camera_left_ : camera_right_);
                eProj -> setId(index);
                eProj->setVertex(0, optimizer.vertex(kf->keyframe_id_ * 4));
                eProj->setVertex(1, (g2o::OptimizableGraph::Vertex *) vXYZ);
                eProj->setMeasurement(Eigen::Vector2d(feat -> position_.pt.x, feat -> position_.pt.y));
                eProj->setInformation(Eigen::Matrix2d::Identity());
                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                eProj->setRobustKernel(rk);
                rk->setDelta(thHuber);
                optimizer.addEdge(eProj);        
                vpEdgePoints.push_back(eProj);
                vpMappoints.push_back(mp);
                vpFeatures.push_back(feat);
                index++;
            }

        }
    }

    optimizer.initializeOptimization();
    optimizer.optimize(100);

    // Check inliers and optimize again without outliers
    int cntPRXYZOutliers = 0;
    for (EdgePRXYZ *e: vpEdgePoints) {
        if (e->chi2() > thHuber2 || e->isDepthValid() == false) {
            e->setLevel(1);
            cntPRXYZOutliers++;
        } else {
            e->setLevel(0);
        }
        e->setRobustKernel(nullptr);
    }
    LOG(INFO) << "Backend::LocalInertialBA(): PRXYZ Outlier/Inlier in optimization: " << cntPRXYZOutliers << "/" << vpEdgePoints.size() - cntPRXYZOutliers;

    // do it again
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    // check Edge PRV
    int cntPRVOutliers = 0;
    for (EdgePRV *e: vpEdgePRV) {
        //LOG(INFO) << "PRV chi2 = " << e->chi2() ;
        if (e->chi2() > thHuberPRV) {
            cntPRVOutliers++;
        }
    }
    LOG(INFO) << "PRV outliers: " << cntPRVOutliers ;

    // recover the pose and points estimation
    for (auto& [_, frame] : keyframes) {
        VertexPose *vPR = (VertexPose *) optimizer.vertex(frame->keyframe_id_ * 4);
        VertexVelocity *vSpeed = (VertexVelocity *) optimizer.vertex(frame->keyframe_id_ * 4 + 1);
        VertexGyroBias *vBg = (VertexGyroBias *) optimizer.vertex(frame->keyframe_id_ * 4 + 2);
        VertexAccBias *vBa = (VertexAccBias *) optimizer.vertex(frame->keyframe_id_ * 4 + 3);

        frame->SetPoseFromIMU(SE3d(vPR->R(), vPR->t()));
        frame->SetVelocitynBias(vSpeed->estimate(), vBg->estimate(), vBa->estimate());
        //frame->ReComputeIMUPreIntegration();
    }
    for (auto& [_, frame] : keyframes) {
        if (_ == 0) continue;
        frame->ReComputeIMUPreIntegration();
    }
    // and the points
    for (auto& [_, mp] : mappoints) {
        if (mp && mp->is_outlier_ == false && mp->observed_times_ > 1) {
            VertexSBAPointXYZ *v = (VertexSBAPointXYZ *) optimizer.vertex(mp->id_ + maxKFid + 1);
            mp->SetPos(v->estimate());
        }
    }

    int cnt_outlier = 0;
    for (size_t i = 0, iend = vpEdgePoints.size(); i < iend; i++) {
        EdgePRXYZ *e = vpEdgePoints[i];
        shared_ptr<MapPoint> mp = vpMappoints[i];
        shared_ptr<Feature>& feat = vpFeatures[i];
        if (e->chi2() > thHuber2 || e->isDepthValid() == false) {
            feat->is_outlier_ = true;
            cnt_outlier++;
            feat->mappoint_.lock()->RemoveObservation(feat);
            feat->mappoint_.reset();
        }
    }
    LOG(INFO) << "Set total " << cnt_outlier << vpEdgePoints.size() - cnt_outlier << " bad map points" << endl;
}


} // namespace demoam