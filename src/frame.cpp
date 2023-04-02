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
}