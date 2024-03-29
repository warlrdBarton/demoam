#ifndef DEMOAM__ALGORITHM_H
#define DEMOAM__ALGORITHM_H

#include "common_include.h"

namespace demoam {

inline bool TriangulatePoints(const VecSE3d& poses,
                              const VecVector3d& pcs,
                              Eigen::Vector3d& pw) {
    Eigen::MatrixX4d A(2 * poses.size(), 4);
    for (size_t i = 0; i < poses.size(); ++i) {
        Eigen::Matrix<double, 3, 4> m = poses[i].matrix3x4();
        A.row(2 * i) = pcs[i][0] * m.row(2) - m.row(0);
        A.row(2 * i + 1) = pcs[i][1] * m.row(2) - m.row(1);
    }

    auto svd = A.jacobiSvd(Eigen::ComputeFullV);
    pw = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();  

    if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-2 && pw[2] > 0) {
        return true; 
    }
    return false; 
}

inline Eigen::Matrix3d NormalizeRotation(const Eigen::Matrix3d &R)
{
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.matrixU() * svd.matrixV().transpose();
}

} // namespace demoam

#endif // DEMOAM__ALGORITHM_H