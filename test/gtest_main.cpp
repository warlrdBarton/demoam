#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include "visual_odometry.h"

DEFINE_string(config_file, "/home/john/demoam/config/default.yaml", "config file path");

int main(int argc, char **argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    std::shared_ptr<demoam::VisualOdometry> vo(
        new demoam::VisualOdometry(FLAGS_config_file));
    assert(vo -> Init() == true);
    vo -> Run();
    
/*
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
*/
    return 0;
}