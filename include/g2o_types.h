#ifndef DEMOAM__G2O_TYPES_H
#define DEMOAM__G2O_TYPES_H

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include "common_include.h"

namespace demoam {

class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW    
    virtual void setToOriginImpl() override { _estimate = Sophus::SE3d();}
    virtual void oplusImpl(const double *update) override {
        Eigen::Vector<double, 6> update_eigen;
        update_eigen << update[0], update[1], update[2],  
                        update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
    }
    virtual bool read(std::istream& is) {return true;}
    virtual bool write(std::ostream& os) const {return true;}
};

class VertexXYZ : public g2o::BaseVertex<3, Eigen::Vector3d> {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW    
    virtual void setToOriginImpl() override { _estimate = Eigen::Vector3d::Zero();}
    virtual void oplusImpl(const double *update) override {
        for (int i = 0; i < 3; ++i) {
            _estimate[i] += update[i];
        }
    }
    virtual bool read(std::istream& is) {return true;}
    virtual bool write(std::ostream& os) const {return true;}
};

class EdgeProjectionPoseOnly : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW    
    EdgeProjectionPoseOnly(const Eigen::Vector3d& pos, const Eigen::Matrix3d& K) 
        : _pw(pos), _K(K) {}
    
    virtual void computeError() override {
        const VertexPose *v = static_cast<const VertexPose *>(_vertices[0]);
        Sophus::SE3d T = v -> estimate();
        Eigen::Vector3d pc = _K * (T * _pw);
        Eigen::Vector2d px = (pc / pc[2]).head<2>();
        _error = _measurement - px;
    }

    virtual void linearizeOplus() override {
        const VertexPose *v = static_cast<const VertexPose *>(_vertices[0]);
        Sophus::SE3d T = v -> estimate();
        Eigen::Vector3d pc = T * _pw;
        double fx = _K(0, 0), fy = _K(1, 1), 
               X = pc[0], Y = pc[1], Z = pc[2], 
               Zinv = 1.0 / (Z + 1e-18), Zinv2 = Zinv * Zinv; 
        _jacobianOplusXi << -fx * Zinv, 0, fx * X * Zinv2, fx * X * Y * Zinv2,
                            -fx - fx * X * X * Zinv2, fx * Y * Zinv, 0, -fy * Zinv,
                            fy * Y * Zinv2, fy + fy * Y * Y * Zinv2, -fy * X * Y * Zinv2,
                            -fy * X * Zinv;
    }
    virtual bool read(std::istream& is) override {return true;}
    virtual bool write(std::ostream& os) const override {return true;}

 private:
    Eigen::Vector3d _pw;
    Eigen::Matrix3d _K;
};

class EdgeProjection : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexPose, VertexXYZ> {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW    
    EdgeProjection(const Eigen::Matrix3d& K, const Sophus::SE3d& camera_pose)
        : _K(K), _camera_pose(camera_pose) {}
    
    virtual void computeError() override {
        const VertexPose* v0 = static_cast<const VertexPose*>(_vertices[0]);
        const VertexXYZ* v1 = static_cast<const VertexXYZ*>(_vertices[1]);
        Sophus::SE3d T = v0 -> estimate();
        Eigen::Vector3d pc = _K * (_camera_pose * (T * v1 -> estimate()));
        Eigen::Vector2d px = (pc / pc[2]).head<2>();
        _error = _measurement - px;
    }

    virtual void linearizeOplus() override {
        const VertexPose* v0 = static_cast<const VertexPose*>(_vertices[0]);
        const VertexXYZ* v1 = static_cast<const VertexXYZ*>(_vertices[1]);
        Sophus::SE3d T = v0 -> estimate();
        Eigen::Vector3d pw = v1 -> estimate();
        Eigen::Vector3d pc = _camera_pose * T * pw;
        double fx = _K(0, 0), fy = _K(1, 1), 
               X = pc[0], Y = pc[1], Z = pc[2], 
               Zinv = 1.0 / (Z + 1e-18), Zinv2 = Zinv * Zinv; 
        _jacobianOplusXi << -fx * Zinv, 0, fx * X * Zinv2, fx * X * Y * Zinv2,
                            -fx - fx * X * X * Zinv2, fx * Y * Zinv, 0, -fy * Zinv,
                            fy * Y * Zinv2, fy + fy * Y * Y * Zinv2, -fy * X * Y * Zinv2,
                            -fy * X * Zinv;
        _jacobianOplusXj = _jacobianOplusXi.block<2, 3>(0, 0) * _camera_pose.rotationMatrix() * T.rotationMatrix();
    }

    virtual bool read(std::istream& is) override {return true;}
    virtual bool write(std::ostream& os) const override {return true;}
 private:
    Eigen::Matrix3d _K;
    Sophus::SE3d _camera_pose;
    
};

} // namespace demoam

#endif // G2O_TYPES_H