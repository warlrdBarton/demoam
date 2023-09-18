#ifndef DEMOAM__VIEWER_H
#define DEMOAM__VIEWER_H

#include <thread>
#include <pangolin/pangolin.h>
#include "common_include.h"

namespace demoam {

class Map;
class Frame;
class MapPoint;

class Viewer {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Viewer();

    void SetMap(std::shared_ptr<Map> map) { map_ = map; }

    void Close();

    void AddCurrentFrame(std::shared_ptr<Frame> current_frame);

    void UpdateMap();

 private:
    void ThreadLoop();

    void DrawFrame(std::shared_ptr<Frame> frame, const float* color);

    void DrawMapPoints();

    void FollowCurrentFrame(pangolin::OpenGlRenderState& vis_camera);

    cv::Mat PlotFrameImage();
    
    std::shared_ptr<Frame>current_frame_ = nullptr;
    std::shared_ptr<Map> map_ = nullptr;

    std::thread viewer_thread_;
    bool viewer_running_ = true;

    std::map<unsigned long, std::shared_ptr<Frame>> active_keyframes_;
    std::unordered_map<unsigned long, std::shared_ptr<MapPoint>> active_mappoints_;
    
    bool map_updated_ = false;

    std::mutex viewer_data_mutex_;
};

} // namespace demoam

#endif // DEMOAM__VIEWER_H
