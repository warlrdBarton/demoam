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
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/base_multi_edge.h>
#include "common_include.h"
#include "config.h"

namespace demoam {

/* Vertices */

class VertexSE3Expmap : public g2o::BaseVertex<6, Sophus::SE3d> {
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

class VertexSBAPointXYZ : public g2o::BaseVertex<3, Eigen::Vector3d> {
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

// Velocity
class VertexVelocity : public g2o::BaseVertex<3, Eigen::Vector3d> {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexVelocity() {}
    
    virtual void setToOriginImpl(){_estimate.setZero(); }
    virtual void oplusImpl(const double *update_) {
        _estimate += Eigen::Map<const Vector3d>(update_);
    }
    virtual bool read(std::istream &is) { return false; }
    virtual bool write(std::ostream &os) const { return false; }
};

// Bias Acc
class VertexAccBias : public VertexVelocity {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexAccBias() {}
};

// Bias Gyro
class VertexGyroBias : public VertexVelocity {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexGyroBias() {}
};

// Gravity direction
class VertexGravityW : public g2o::BaseVertex<2, Eigen::Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VertexGravityW(){}

    bool read(std::istream &is) { return true; }

    bool write(std::ostream &os) const { return true; }

    virtual void setToOriginImpl() {
        _estimate = Vector3d(0, 0, settings::GRAVITY_VALUE);
    }

    virtual void oplusImpl(const double *update_) {
        _estimate = SO3d::exp(Vector3d(update_[0], update_[1], 0)) * _estimate;
    }
};

// VIO Pose
/**
 * 旋转在前的SO3+t类型pose，6自由度，存储时伪装为g2o::VertexSE3，供g2o_viewer查看
 */
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexPose() {}
    virtual void setToOriginImpl() override {
        _estimate = Sophus::SE3d();
    }
    virtual void oplusImpl(const double *update_) override {
        _estimate.so3() = _estimate.so3() * SO3d::exp(Eigen::Map<const Vector3d>(&update_[0])); // Rotation
        _estimate.translation() += Eigen::Map<const Vector3d>(&update_[3]); // Translation
    }
    bool read(std::istream &is)  { return true; }
    bool write(std::ostream &os) const  { return true; }
};

/*********************************************************************************************************************************************/
/* Edges */
/*********************************************************************************************************************************************/

class EdgeSE3ProjectXYZPoseOnly : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexSE3Expmap> {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW    
    EdgeSE3ProjectXYZPoseOnly(const Eigen::Vector3d& pos, const Eigen::Matrix3d& K, const Sophus::SE3d& camera_pose) 
        : _pw(pos), _K(K), _camera_pose(camera_pose) {}
    
    virtual void computeError() override {
        const VertexSE3Expmap *v = static_cast<const VertexSE3Expmap *>(_vertices[0]);
        Sophus::SE3d T = v -> estimate();
        Eigen::Vector3d pc = _K * (_camera_pose *(T * _pw));
        Eigen::Vector2d px = (pc / pc[2]).head<2>();
        _error = _measurement - px;
    }

    virtual void linearizeOplus() override {
        const VertexSE3Expmap *v = static_cast<const VertexSE3Expmap *>(_vertices[0]);
        Sophus::SE3d T = v -> estimate();
        Eigen::Vector3d pc = _camera_pose * T * _pw;
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
    Sophus::SE3d _camera_pose;
};

class EdgeSE3ProjectXYZ : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexSE3Expmap, VertexSBAPointXYZ> {
    
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW    
    EdgeSE3ProjectXYZ(const Eigen::Matrix3d& K, const Sophus::SE3d& camera_pose)
        : _K(K), _camera_pose(camera_pose) {}
    
    virtual void computeError() override {
        const VertexSE3Expmap* v0 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
        const VertexSBAPointXYZ* v1 = static_cast<const VertexSBAPointXYZ*>(_vertices[1]);
        Sophus::SE3d T = v0 -> estimate();
        Eigen::Vector3d pc = _K * (_camera_pose * (T * v1 -> estimate()));
        Eigen::Vector2d px = (pc / pc[2]).head<2>();
        _error = _measurement - px;
    }

    virtual void linearizeOplus() override {
        const VertexSE3Expmap* v0 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
        const VertexSBAPointXYZ* v1 = static_cast<const VertexSBAPointXYZ*>(_vertices[1]);
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

class EdgeGyroBias : public g2o::BaseUnaryEdge<3, Vector3d, VertexGyroBias> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeGyroBias() : BaseUnaryEdge<3, Vector3d, VertexGyroBias>() {}

    Matrix3d dRbij;
    Matrix3d J_dR_bg;
    Matrix3d Rwbi;
    Matrix3d Rwbj;

    virtual void computeError() override {
        const VertexGyroBias *v = static_cast<const VertexGyroBias *>(_vertices[0]);
        Vector3d bg = v->estimate();
        Matrix3d dRbg = SO3d::exp(J_dR_bg * bg).matrix();
        SO3d errR((dRbij * dRbg).transpose() * Rwbi.transpose() * Rwbj); // dRij^T * Riw * Rwj
        _error = errR.log();
    }

    virtual void linearizeOplus() override {
        SO3d errR(dRbij.transpose() * Rwbi.transpose() * Rwbj); // dRij^T * Riw * Rwj
        Matrix3d Jlinv = SO3d::jl_inv(errR.log());

        _jacobianOplusXi = -Jlinv * J_dR_bg;
    }

    bool read(std::istream &is)  { return true; }

    bool write(std::ostream &os) const  { return true; }
};




} // namespace demoam

#endif // G2O_TYPES_H