#pragma once

#include "common_include.h"

namespace demoam {

inline bool TriangulatePoints(const std::vector<Sophus::SE3d>& poses,
                              const std::vector<Eigen::Vector3d>& pcs,
                              Eigen::Vector3d& pw) {

    Eigen::MatrixX4d A(2 * poses.size(), 4);
    for (size_t i = 0; i < poses.size(); ++i) {
        Eigen::Matrix<double, 3, 4> m = poses[i].matrix3x4();
        A.row(2 * i) = pcs[i][0] * m.row(2) - m.row(0);
        A.row(2 * i + 1) = pcs[i][1] * m.row(2) - m.row(1);
    }

    auto svd = A.jacobiSvd(Eigen::ComputeFullV);
    pw = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();  

    if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-2) {
        return true;
    }
    return false;
}

}