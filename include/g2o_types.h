#ifndef DEMOAM__G2O_TYPES_H
#define DEMOAM__G2O_TYPES_H

#include <g2o/core/base_multi_edge.h>
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

#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <g2o/types/slam3d/edge_pointxyz.h>

#include "common_include.h"
#include "config.h"
#include "camera.h"
#include "imu_types.h"

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

// world cordinate(camera0)
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
typedef g2o::VertexPointXYZ VertexVelocity;

// Bias Acc
typedef g2o::VertexPointXYZ VertexAccBias;

// Bias Gyro
typedef g2o::VertexPointXYZ VertexGyroBias;

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
 * IMUPose, Twb
 * 参数化为P+R，右乘更新，P在前，6自由度，存储时伪装为g2o::VertexSE3，供g2o_viewer查看
 */
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexPose() {}
    virtual void setToOriginImpl() override {
        _estimate = Sophus::SE3d();
    }
    virtual void oplusImpl(const double *update_) override {
        _estimate.so3() = _estimate.so3() * SO3d::exp(Eigen::Map<const Vector3d>(&update_[3])); // Rotation
        _estimate.translation() += Eigen::Map<const Vector3d>(&update_[0]); // Translation
    }
    bool read(std::istream &is)  { return true; }
    bool write(std::ostream &os) const  { return true; }
    Matrix3d R() const {
        return _estimate.so3().matrix();
    }
    Vector3d t() const {
        return _estimate.translation().matrix();
    }
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

/**
 * @brief The EdgeGyroBias class
 * For gyroscope bias compuation in Visual-Inertial initialization
 */
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

/**
 * The pre-integration IMU motion constraint
 * Connect 6 vertex: PR0, V0, biasG0, bias A0 and PR1, V1
 * Vertex 0: PR0
 * Vertex 1: PR1
 * Vertex 2: V0
 * Vertex 3: V1
 * Vertex 4: biasG0
 * Vertex 5: biasA0
 * Error order: error_P, error_R, error_V
 */
class EdgePRV : public g2o::BaseMultiEdge<9, IMUPreIntegration> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgePRV(const Vector3d &gw) : g2o::BaseMultiEdge<9, IMUPreIntegration>(), GravityVec(gw) {
        resize(6);
    }

    bool read(std::istream &is) override { return true; }

    bool write(std::ostream &os) const override { return true; }

    virtual void computeError() override {
        const VertexPose *vPRi = static_cast<const VertexPose *>(_vertices[0]);
        const VertexPose *vPRj = static_cast<const VertexPose *>(_vertices[1]);
        const VertexVelocity *vVi = static_cast<const VertexVelocity *>(_vertices[2]);
        const VertexVelocity *vVj = static_cast<const VertexVelocity *>(_vertices[3]);
        const VertexGyroBias *vBiasGi = static_cast<const VertexGyroBias *>(_vertices[4]);
        const VertexAccBias *vBiasAi = static_cast<const VertexAccBias *>(_vertices[5]);

        // terms need to computer error in vertex i, except for bias error
        const Vector3d Pi = vPRi->t();
        const Matrix3d Ri = vPRi->R();

        const Vector3d Vi = vVi->estimate();

        // Bias from the bias vertex
        const Vector3d dBgi = vBiasGi->estimate();
        const Vector3d dBai = vBiasAi->estimate();

        // terms need to computer error in vertex j, except for bias error
        const Vector3d Pj = vPRj->t();
        const Matrix3d Rj = vPRj->R();

        const Vector3d Vj = vVj->estimate();

        // IMU Preintegration measurement
        const IMUPreIntegration &M = _measurement;

        const double dTij = M.getDeltaTime();   // Delta Time
        const double dT2 = dTij * dTij;
        const Vector3d dPij = M.getDeltaP().cast<double>();    // Delta Position pre-integration measurement
        const Vector3d dVij = M.getDeltaV().cast<double>();    // Delta Velocity pre-integration measurement
        const Matrix3d dRij = M.getDeltaR().cast<double>();    // Delta Rotation pre-integration measurement

        // tmp variable, transpose of Ri
        const Matrix3d RiT = Ri.inverse();
        // residual error of Delta Position measurement
        const Vector3d rPij = RiT * (Pj - Pi - Vi * dTij - 0.5 * GravityVec * dT2)
                              - (dPij + M.getJPBiasg() * dBgi +
                                 M.getJPBiasa() * dBai);   // this line includes correction term of bias change.

        // residual error of Delta Velocity measurement
        const Vector3d rVij = RiT * (Vj - Vi - GravityVec * dTij)
                              - (dVij + M.getJVBiasg() * dBgi +
                                 M.getJVBiasa() * dBai);   //this line includes correction term of bias change

        // residual error of Delta Rotation measurement
        const Matrix3d dR_dbg = SO3d::exp(M.getJRBiasg() * dBgi).matrix();
        const SO3d rRij((dRij * dR_dbg).inverse() * RiT * Rj);
        const Vector3d rPhiij = rRij.log();

        // 9-Dim error vector order:
        // position-velocity-rotation
        // rPij - rPhiij - rVij
        _error.segment<3>(0) = rPij;       // position error
        _error.segment<3>(3) = rPhiij;     // rotation phi error
        _error.segment<3>(6) = rVij;       // velocity error
    }

    virtual void linearizeOplus() override {
        const VertexPose *vPRi = static_cast<const VertexPose *>(_vertices[0]);
        const VertexPose *vPRj = static_cast<const VertexPose *>(_vertices[1]);
        const VertexVelocity *vVi = static_cast<const VertexVelocity *>(_vertices[2]);
        const VertexVelocity *vVj = static_cast<const VertexVelocity *>(_vertices[3]);
        const VertexGyroBias *vBiasGi = static_cast<const VertexGyroBias *>(_vertices[4]);
        const VertexAccBias *vBiasAi = static_cast<const VertexAccBias *>(_vertices[5]);

        // terms need to computer error in vertex i, except for bias error
        const Vector3d Pi = vPRi->t();
        const Matrix3d Ri = vPRi->R();

        const Vector3d Vi = vVi->estimate();

        // Bias from the bias vertex
        const Vector3d dBgi = vBiasGi->estimate();
        const Vector3d dBai = vBiasAi->estimate();

        // terms need to computer error in vertex j, except for bias error
        const Vector3d Pj = vPRj->t();
        const Matrix3d Rj = vPRj->R();

        const Vector3d Vj = vVj->estimate();

        // IMU Preintegration measurement
        const IMUPreIntegration &M = _measurement;
        const double dTij = M.getDeltaTime();   // Delta Time
        const double dT2 = dTij * dTij;

        // some temp variable
        Matrix3d O3x3 = Matrix3d::Zero();       // 0_3x3
        Matrix3d RiT = Ri.transpose();          // Ri^T
        Matrix3d RjT = Rj.transpose();          // Rj^T
        Vector3d rPhiij = _error.segment<3>(3); // residual of rotation, rPhiij
        Matrix3d JrInv_rPhi = SO3d::jr_inv(rPhiij);    // inverse right jacobian of so3 term #rPhiij#
        Matrix3d J_rPhi_dbg = M.getJRBiasg();              // jacobian of preintegrated rotation-angle to gyro bias i

        // this is really messy
        // 1.
        // increment is the same as Forster 15'RSS
        // pi = pi + dpi,    pj = pj + dpj
        // Ri = Ri*Exp(dphi_i), Rj = Rj*Exp(dphi_j)
        // vi = vi + dvi,       vj = vj + dvj
        //      Note: the optimized bias term is the 'delta bias'
        // dBgi = dBgi + dbgi_update,    dBgj = dBgj + dbgj_update
        // dBai = dBai + dbai_update,    dBaj = dBaj + dbaj_update

        // 2.
        // 9-Dim error vector order in PVR:
        // position-velocity-rotation
        // rPij - rPhiij - rVij
        //      Jacobian row order:
        // J_rPij_xxx
        // J_rPhiij_xxx
        // J_rVij_xxx

        // 3.
        // order in 'update_' in PR
        // Vertex_i : dPi, dPhi_i
        // Vertex_j : dPj, dPhi_j
        // 6-Dim error vector order in Bias:
        // dBiasg_i - dBiasa_i

        // Jacobians:
        // dP/dPR0, dP/dPR1, dP/dV0, dP/dV1, dP/dBiasG, dP/dBiasG
        // dR/dPR0, dR/dPR1, dR/dV0, dR/dV1, dR/dBiasG, dR/dBiasG
        // dV/dPR0, dV/dPR1, dV/dV0, dV/dV1, dV/dBiasG, dV/dBiasG

        // 4. PR0 & V0
        // For Vertex_PR_i, J [dP;dR;dV] / [dP0 dR0]
        Eigen::Matrix<double, 9, 6> JPRi;
        JPRi.setZero();
        // J_rPij_xxx_i for Vertex_PR_i
        JPRi.block<3, 3>(0, 0) = -RiT;      //J_rP_dpi
        JPRi.block<3, 3>(0, 3) = SO3d::hat(
                RiT * (Pj - Pi - Vi * dTij - 0.5 * GravityVec * dT2));    //J_rP_dPhi_i
        // J_rPhiij_xxx_i for Vertex_PR_i
        Matrix3d ExprPhiijTrans = SO3d::exp(rPhiij).inverse().matrix();
        Matrix3d JrBiasGCorr = SO3d::jr(J_rPhi_dbg * dBgi);
        JPRi.block<3, 3>(3, 0) = O3x3;    //dpi
        JPRi.block<3, 3>(3, 3) = -JrInv_rPhi * RjT * Ri;    //dphi_i
        // J_rVij_xxx_i for Vertex_PVR_i
        JPRi.block<3, 3>(6, 0) = O3x3;    //dpi
        JPRi.block<3, 3>(6, 3) = SO3d::hat(RiT * (Vj - Vi - GravityVec * dTij));    //dphi_i

        // For Vertex_V_i, J [dP;dR;dV] / dV0
        Eigen::Matrix<double, 9, 3> JVi;
        JVi.setZero();
        JVi.block<3, 3>(0, 0) = -RiT * dTij;  //J_rP_dvi
        JVi.block<3, 3>(3, 0) = O3x3;    //rR_dvi
        JVi.block<3, 3>(6, 0) = -RiT;    //rV_dvi

        // 5. PR1 & V1
        // For Vertex_PR_j, J [dP;dR;dV] / [dP1 dR1]
        Eigen::Matrix<double, 9, 6> JPRj;
        JPRj.setZero();
        // J_rPij_xxx_j for Vertex_PR_j
        JPRj.block<3, 3>(0, 0) = RiT;  //rP_dpj
        JPRj.block<3, 3>(0, 3) = O3x3;    //rP_dphi_j
        // J_rPhiij_xxx_j for Vertex_PR_j
        JPRj.block<3, 3>(3, 0) = O3x3;    //rR_dpj
        JPRj.block<3, 3>(3, 3) = JrInv_rPhi;    //rR_dphi_j
        // J_rVij_xxx_j for Vertex_PR_j
        JPRj.block<3, 3>(6, 0) = O3x3;    //rV_dpj
        JPRj.block<3, 3>(6, 3) = O3x3;    //rV_dphi_j

        // For Vertex_V_i, J [dP;dR;dV] / dV1
        Eigen::Matrix<double, 9, 3> JVj;
        JVj.setZero();
        JVj.block<3, 3>(0, 0) = O3x3;    //rP_dvj
        JVj.block<3, 3>(3, 0) = O3x3;    //rR_dvj
        JVj.block<3, 3>(6, 0) = RiT;    //rV_dvj

        // 6.
        // For Vertex_Bias_i
        Eigen::Matrix<double, 9, 3> JBiasG;
        Eigen::Matrix<double, 9, 3> JBiasA;
        JBiasG.setZero();
        JBiasA.setZero();

        // bias
        JBiasG.block<3, 3>(0, 0) = -M.getJPBiasg();     //J_rP_dbgi
        JBiasG.block<3, 3>(3, 0) = -JrInv_rPhi * ExprPhiijTrans * JrBiasGCorr * J_rPhi_dbg;    //dbg_i
        JBiasG.block<3, 3>(6, 0) = -M.getJVBiasg();    //dbg_i

        JBiasA.block<3, 3>(0, 0) = -M.getJPBiasa();     //J_rP_dbai
        JBiasA.block<3, 3>(3, 0) = O3x3;    //dba_i
        JBiasA.block<3, 3>(6, 0) = -M.getJVBiasa();    //dba_i

        // set all jacobians
        _jacobianOplus[0] = JPRi;
        _jacobianOplus[1] = JPRj;
        _jacobianOplus[2] = JVi;
        _jacobianOplus[3] = JVj;
        _jacobianOplus[4] = JBiasG;
        _jacobianOplus[5] = JBiasA;
    }

protected:
    Vector3d GravityVec;
};

/**
 * XYZ reprojection err
 * Vertex0 PR
 * Vertex1 XYZ
 */
class EdgePRXYZ : public g2o::BaseBinaryEdge<2, Vector2d, VertexPose, VertexSBAPointXYZ> {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgePRXYZ(const shared_ptr<Camera>& cam) : cam_(cam) {}
    bool read(std::istream &is) override { return true; }
    bool write(std::ostream &os) const override { return true; }
    virtual void computeError() override {
        VertexPose *vPR0 = dynamic_cast<VertexPose *>(_vertices[0]);
        VertexSBAPointXYZ *v1 = dynamic_cast<VertexSBAPointXYZ *>(_vertices[1]);

        const Matrix3d Rwb0 = vPR0->R();
        const Vector3d twb0 = vPR0->t();
        SE3d Twc = SE3d(Rwb0, twb0) * settings::Tbc;
        SE3d Tcw = Twc.inverse();

        Vector3d pc = cam_->world2camera(v1->estimate(), Tcw);
        depth_ = pc[2];

        Vector2d px = cam_->camera2pixel(pc);

        _error = _measurement - px;
    }
    virtual void linearizeOplus() override {
        const VertexPose *vPR0 = dynamic_cast<const VertexPose *>(_vertices[0]);
        const VertexSBAPointXYZ *vXYZ = dynamic_cast<const VertexSBAPointXYZ *>(_vertices[1]);

        const Matrix3d Rwb = vPR0->R();
        const Vector3d Pwb = vPR0->t();
        SE3d Twb(Rwb, Pwb);
        const Matrix3d Rcb = settings::Tcb.rotationMatrix();
        const Vector3d Pcb = settings::Tcb.translation();
        SE3d Twc = Twb * settings::Tbc;
        SE3d Tcw = Twc.inverse();
        const Vector3d Pw = vXYZ->estimate();

        // point coordinate in reference KF, body
        Vector3d Pc = cam_->world2camera(Pw, Tcw);

        double x = Pc[0];
        double y = Pc[1];
        double z = Pc[2];
        double zinv = 1.0 / (z + 1e-9);
        double fx = cam_->fx();
        double fy = cam_->fy();

        Matrix<double, 2, 3> Maux;
        Maux.setZero();
        Maux(0, 0) = fx;
        Maux(0, 1) = 0;
        Maux(0, 2) = -x * zinv * fx;
        Maux(1, 0) = 0;
        Maux(1, 1) = fy;
        Maux(1, 2) = -y * zinv * fy;
        Matrix<double, 2, 3> Jpi = Maux / z;

        // Jacobian of Pc/error w.r.t dPwb
        Matrix<double, 2, 3> JdPwb = -Jpi * (-Rcb * Rwb.transpose());
        // Jacobian of Pc/error w.r.t dRwb
        Vector3d Paux = Rcb * Rwb.transpose() * (Pw - Pwb);
        Matrix<double, 2, 3> JdRwb = -Jpi * (SO3d::hat(Paux) * Rcb);

        Matrix<double, 2, 6> JPR = Matrix<double, 2, 6>::Zero();
        JPR.block<2, 3>(0, 0) = JdPwb;
        JPR.block<2, 3>(0, 3) = JdRwb;

        _jacobianOplusXi = JPR;
        _jacobianOplusXj = Jpi * Rcb * Rwb.transpose();
    }
    bool isDepthValid() {
        return depth_ > 0;
    }
protected:
    double depth_ = 0;
    shared_ptr<Camera> cam_ = nullptr;
};

typedef g2o::EdgePointXYZ EdgeBiasG;
typedef g2o::EdgePointXYZ EdgeBiasA;

} // namespace demoam

#endif // G2O_TYPES_H