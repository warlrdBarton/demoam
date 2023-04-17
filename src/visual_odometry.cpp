#include "visual_odometry.h"
#include "config.h"
#include "dataset.h"
#include "frontend.h"
#include "backend.h"
#include "map.h"
#include "frame.h"
#include "viewer.h"

//#define TEST

namespace demoam {
VisualOdometry::VisualOdometry(std::string config_path) 
    : config_file_path_(config_path) {}

bool VisualOdometry::Init() {
    if (Config::SetParameterFile(config_file_path_) == false) {
        return false;
    }
    dataset_ = std::shared_ptr<Dataset> (new Dataset(Config::Get<std::string>("dataset_dir")));
    CHECK_EQ(dataset_ -> Init(), true);

    frontend_ = std::shared_ptr<Frontend> (new Frontend);
    backend_ = std::shared_ptr<Backend> (new Backend);
    map_ = std::shared_ptr<Map> (new Map);
    viewer_ = std::shared_ptr<Viewer> (new Viewer);

    frontend_ -> SetBackend(backend_);
    frontend_ -> SetMap(map_);
    frontend_ -> SetViewer(viewer_);
    frontend_ -> SetCameras(dataset_ -> GetCamera(0), dataset_ -> GetCamera(1));

    backend_ -> SetMap(map_);
    backend_ -> SetCameras(dataset_ -> GetCamera(0), dataset_ -> GetCamera(1));

    viewer_->SetMap(map_);
    return true;
}

void VisualOdometry::Run() {
    LOG(INFO) << "VisualOdometry::Run(): Visual Odometry starts...";
    #ifdef TEST
        int iter = 0;
    #endif
    while (true) {
        if (!VisualOdometry::Step()) break;
        LOG(INFO) << "\n\n\n";
        #ifdef TEST
            if (iter++ > 400) break;
        #endif
    }
    frontend_ -> Stop();
    backend_ -> Stop();
    viewer_ -> Close();
    LOG(INFO) << "VisualOdometry::Run(): Visual Odometry exits";   
}

bool VisualOdometry::Step() {
    std::shared_ptr<Frame> new_frame = dataset_ -> NextFrame();
    if (!new_frame) return false;
    auto t1 = std::chrono::steady_clock::now();
    bool success = frontend_ -> AddFrame(new_frame);
    auto t2 = std::chrono::steady_clock::now();
    auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    LOG(INFO) << "VisualOdometry::Step(): VO cost time: " << time_used.count() << " seconds.";
    return success;
}

}