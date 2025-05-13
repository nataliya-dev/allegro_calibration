//
// Created by davide on 09/10/23.
//

#ifndef METRIC_CALIBRATOR_HANDEYE_CALIBRATOR_H
#define METRIC_CALIBRATOR_HANDEYE_CALIBRATOR_H

#include "Eigen/Core"
#include "ceres/ceres.h"
#include <ceres/rotation.h>

#include "PinholeCameraModel.h"

struct CalibReprojectionError
{
    CalibReprojectionError( const PinholeCameraModel &cam_model,
                            const Eigen::Vector3d &robot_r_vec,
                            const Eigen::Vector3d &robot_t_vec,
                            const Eigen::Vector3d &pattern_pt,
                            const Eigen::Vector2d &observed_pt ) :
            cam_model(cam_model),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec),
            pattern_pt(pattern_pt),
            observed_pt( observed_pt) {}

    //template <typename T>
    bool operator()(//const double* const weights,
            const double* const board2ee,
            const double* const h2e,
            double* residuals) const
    {
        //T ptn_pt[3] = {T(pattern_pt(0)), T(pattern_pt(1)), T(pattern_pt(2))},
        //  tcp_pt[3], base_r[3], cam_pt[3];
        double ptn_pt[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))},
                tcp_pt[3], base_r[3], cam_pt[3];

        // Apply the current board2ee transformation
        ceres::AngleAxisRotatePoint(board2ee, ptn_pt, tcp_pt);

        tcp_pt[0] += board2ee[3];
        tcp_pt[1] += board2ee[4];
        tcp_pt[2] += board2ee[5];

        double robot_r[3] = {double(robot_r_vec(0)), double(robot_r_vec(1)), double(robot_r_vec(2))},
               robot_t[3] = {double(robot_t_vec(0)), double(robot_t_vec(1)), double(robot_t_vec(2))};

        // Apply the robot position transformation
        ceres::AngleAxisRotatePoint(robot_r, tcp_pt, base_r);

        base_r[0] += robot_t[0];
        base_r[1] += robot_t[1];
        base_r[2] += robot_t[2];

        // Apply the current h2e transformation
        ceres::AngleAxisRotatePoint(h2e, base_r, cam_pt);

        cam_pt[0] += h2e[3];
        cam_pt[1] += h2e[4];
        cam_pt[2] += h2e[5];


        // Projection.
        double proj_pt[2];
        cam_model.project(cam_pt, proj_pt);

        residuals[0] = (proj_pt[0] - double(observed_pt(0)));
        residuals[1] = (proj_pt[1] - double(observed_pt(1)));
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const PinholeCameraModel &cam_model,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec,
                                        const Eigen::Vector3d &pattern_pt,
                                        const Eigen::Vector2d &observed_pt )
    {
        return (new ceres::NumericDiffCostFunction<CalibReprojectionError, ceres::RIDDERS,  2, 6, 6>(
                new CalibReprojectionError( cam_model, robot_r_vec, robot_t_vec, pattern_pt, observed_pt )));
    }

    const PinholeCameraModel &cam_model;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
    Eigen::Vector3d pattern_pt;
    Eigen::Vector2d observed_pt;
};


struct EyeInHandCalibReprojectionError
{
    EyeInHandCalibReprojectionError(const PinholeCameraModel &cam_model,
                                    const Eigen::Vector3d &robot_r_vec,
                                    const Eigen::Vector3d &robot_t_vec,
                                    const Eigen::Vector3d &pattern_pt,
                                    const Eigen::Vector2d &observed_pt ) :
            cam_model(cam_model),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec),
            pattern_pt(pattern_pt),
            observed_pt(observed_pt) {}

    bool operator()(
            const double* const b2w,
            const double* const h2e,
            double* residuals) const
    {
        double ptn_pt[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))},
                tcp_pt[3], base_r[3], cam_pt[3], rot_mat_h2e[9], rot_mat_h2e_inv[9], h2e_vec_inv[3], h2e_tras_vec_inv[3];

        // Apply the current h2e transformation
        ceres::AngleAxisRotatePoint(b2w, ptn_pt, base_r);

        base_r[0] += b2w[3];
        base_r[1] += b2w[4];
        base_r[2] += b2w[5];

        double robot_r[3] = {double(robot_r_vec(0)), double(robot_r_vec(1)), double(robot_r_vec(2))},
          robot_t[3] = {double(robot_t_vec(0)), double(robot_t_vec(1)), double(robot_t_vec(2))};

        // Apply the robot position transformation
        ceres::AngleAxisRotatePoint(robot_r, base_r, tcp_pt);

        tcp_pt[0] += robot_t[0];
        tcp_pt[1] += robot_t[1];
        tcp_pt[2] += robot_t[2];

        // Apply the current h2e transformation
        ceres::AngleAxisToRotationMatrix(h2e, rot_mat_h2e);
        cv::Mat h2e_mat = (cv::Mat_<double>(4,4) <<
                rot_mat_h2e[0], rot_mat_h2e[3], rot_mat_h2e[6], h2e[3],
                rot_mat_h2e[1], rot_mat_h2e[4], rot_mat_h2e[7], h2e[4],
                rot_mat_h2e[2], rot_mat_h2e[5], rot_mat_h2e[8], h2e[5],
                0, 0, 0, 1);

        cv::Mat h2e_mat_inv = h2e_mat.inv();

        rot_mat_h2e_inv[0] = h2e_mat_inv.at<double>(0,0);
        rot_mat_h2e_inv[1] = h2e_mat_inv.at<double>(1,0);
        rot_mat_h2e_inv[2] = h2e_mat_inv.at<double>(2,0);
        rot_mat_h2e_inv[3] = h2e_mat_inv.at<double>(0,1);
        rot_mat_h2e_inv[4] = h2e_mat_inv.at<double>(1,1);
        rot_mat_h2e_inv[5] = h2e_mat_inv.at<double>(2,1);
        rot_mat_h2e_inv[6] = h2e_mat_inv.at<double>(0,2);
        rot_mat_h2e_inv[7] = h2e_mat_inv.at<double>(1,2);
        rot_mat_h2e_inv[8] = h2e_mat_inv.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_h2e_inv, h2e_vec_inv);

        h2e_tras_vec_inv[0] = h2e_mat_inv.at<double>(0,3);
        h2e_tras_vec_inv[1] = h2e_mat_inv.at<double>(1,3);
        h2e_tras_vec_inv[2] = h2e_mat_inv.at<double>(2,3);

        ceres::AngleAxisRotatePoint(h2e_vec_inv, tcp_pt, cam_pt);

        cam_pt[0] += h2e_tras_vec_inv[0];
        cam_pt[1] += h2e_tras_vec_inv[1];
        cam_pt[2] += h2e_tras_vec_inv[2];


        // Projection.
        double proj_pt[2];
        cam_model.project(cam_pt, proj_pt);

        residuals[0] = (proj_pt[0] - double(observed_pt(0)));
        residuals[1] = (proj_pt[1] - double(observed_pt(1)));
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const PinholeCameraModel &cam_model,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec,
                                        const Eigen::Vector3d &pattern_pt,
                                        const Eigen::Vector2d &observed_pt )
    {
        return (new ceres::NumericDiffCostFunction<EyeInHandCalibReprojectionError, ceres::CENTRAL, 2, 6, 6>(
                new EyeInHandCalibReprojectionError( cam_model, robot_r_vec, robot_t_vec, pattern_pt, observed_pt )));
    }

    const PinholeCameraModel &cam_model;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
    Eigen::Vector3d pattern_pt;
    Eigen::Vector2d observed_pt;
};




struct MobileCalibReprojectionError
{
    MobileCalibReprojectionError( const PinholeCameraModel &cam_model,
                                  const Eigen::Vector3d &robot_r_vec,
                                  const Eigen::Vector3d &robot_t_vec,
                                     const Eigen::Vector3d &pattern_pt,
                                     const Eigen::Vector2d &observed_pt ) :
            cam_model(cam_model),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec),
            pattern_pt(pattern_pt),
            observed_pt(observed_pt) {}

    template <typename T>
    bool operator()(
            const T* const b2w,
            const T* const h2e,
            const T* const robot_opt,
            T* residuals) const
    {
        T ptn_pt[3] = {T(pattern_pt(0)), T(pattern_pt(1)), T(pattern_pt(2))},
                tcp_pt[3], base_r[3], cam_pt[3], tcp_pt_opt[3], rot_mat[9], rot_vec[3], rot_mat_opt[9], tras_vec_opt[3];

        // Apply the current h2e transformation
        ceres::AngleAxisRotatePoint(b2w, ptn_pt, base_r);

        base_r[0] += b2w[3];
        base_r[1] += b2w[4];
        base_r[2] += b2w[5];

        T robot_r[3] = {T(robot_r_vec(0)), T(robot_r_vec(1)), T(robot_r_vec(2))},
                robot_t[3] = {T(robot_t_vec(0)), T(robot_t_vec(1)), T(robot_t_vec(2))};

        // Apply the robot position transformation
        ceres::AngleAxisRotatePoint(robot_r, base_r, tcp_pt);

        tcp_pt[0] += robot_t[0];
        tcp_pt[1] += robot_t[1];
        tcp_pt[2] += robot_t[2];

        // Apply the robot position transformation
        ceres::AngleAxisRotatePoint(robot_opt, tcp_pt, tcp_pt_opt);

        tcp_pt_opt[0] += robot_opt[3];
        tcp_pt_opt[1] += robot_opt[4];
        tcp_pt_opt[2] += robot_opt[5];

        // Apply the current h2e transformation
        ceres::AngleAxisRotatePoint(h2e, tcp_pt_opt, cam_pt);

        cam_pt[0] += h2e[3];
        cam_pt[1] += h2e[4];
        cam_pt[2] += h2e[5];


        // Projection.
        T proj_pt[2];
        cam_model.project(cam_pt, proj_pt);

        residuals[0] = (proj_pt[0] - T(observed_pt(0)));
        residuals[1] = (proj_pt[1] - T(observed_pt(1)));
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const PinholeCameraModel &cam_model,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec,
                                        const Eigen::Vector3d &pattern_pt,
                                        const Eigen::Vector2d &observed_pt )
    {
        return (new ceres::AutoDiffCostFunction<MobileCalibReprojectionError, 2, 6, 6, 6>(
                new MobileCalibReprojectionError( cam_model, robot_r_vec, robot_t_vec, pattern_pt, observed_pt )));
    }

    const PinholeCameraModel &cam_model;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
    Eigen::Vector3d pattern_pt;
    Eigen::Vector2d observed_pt;
};



struct MobileEyeInHandCalibReprojectionError
{
    MobileEyeInHandCalibReprojectionError( const PinholeCameraModel &cam_model,
                                           const Eigen::Vector3d &robot_r_vec,
                                           const Eigen::Vector3d &robot_t_vec,
                                          const Eigen::Vector3d &pattern_pt,
                                          const Eigen::Vector2d &observed_pt ) :
            cam_model(cam_model),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec),
            pattern_pt(pattern_pt),
            observed_pt( observed_pt) {}

    //template <typename T>
    bool operator()(
            const double* const b2w,
            const double* const h2e,
            const double* const robot_opt,
            double* residuals) const
    {
        double ptn_pt[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))},
                tcp_pt[3], base_r[3], cam_pt[3], opt_tcp_pt[3];

        // Apply the current h2e transformation
        ceres::AngleAxisRotatePoint(b2w, ptn_pt, base_r);

        base_r[0] += b2w[3];
        base_r[1] += b2w[4];
        base_r[2] += b2w[5];

        double robot_r[3] = {double(robot_r_vec(0)), double(robot_r_vec(1)), double(robot_r_vec(2))},
                robot_t[3] = {double(robot_t_vec(0)), double(robot_t_vec(1)), double(robot_t_vec(2))};

        // Apply the robot position transformation
        ceres::AngleAxisRotatePoint(robot_r, base_r, tcp_pt);

        tcp_pt[0] += robot_t[0];
        tcp_pt[1] += robot_t[1];
        tcp_pt[2] += robot_t[2];

        double robot_r_opt[3] = {double(robot_opt[0]), double(robot_opt[1]), double(robot_opt[2])},
                robot_t_opt[3] = {double(robot_opt[3]), double(robot_opt[4]), double(robot_opt[5])};

        // Apply the robot position transformation
        ceres::AngleAxisRotatePoint(robot_r_opt, tcp_pt, opt_tcp_pt);

        opt_tcp_pt[0] += robot_t_opt[0];
        opt_tcp_pt[1] += robot_t_opt[1];
        opt_tcp_pt[2] += robot_t_opt[2];

        // Apply the current h2e transformation
        ceres::AngleAxisRotatePoint(h2e, opt_tcp_pt, cam_pt);

        cam_pt[0] += h2e[3];
        cam_pt[1] += h2e[4];
        cam_pt[2] += h2e[5];


        // Projection.
        double proj_pt[2];
        cam_model.project(cam_pt, proj_pt);

        residuals[0] = (proj_pt[0] - double(observed_pt(0)));
        residuals[1] = (proj_pt[1] - double(observed_pt(1)));
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const PinholeCameraModel &cam_model,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec,
                                        const Eigen::Vector3d &pattern_pt,
                                        const Eigen::Vector2d &observed_pt )
    {
        return (new ceres::NumericDiffCostFunction<MobileEyeInHandCalibReprojectionError, ceres::CENTRAL,  2, 6, 6, 6>(
                new MobileEyeInHandCalibReprojectionError( cam_model, robot_r_vec, robot_t_vec, pattern_pt, observed_pt )));
    }

    const PinholeCameraModel &cam_model;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
    Eigen::Vector3d pattern_pt;
    Eigen::Vector2d observed_pt;
};


struct MultiCalibReprojectionError
{
    MultiCalibReprojectionError( const PinholeCameraModel &cam_model,
                                 const Eigen::Vector3d &robot_r_vec,
                                 const Eigen::Vector3d &robot_t_vec,
                                 const Eigen::Vector3d &pattern_pt,
                                 const Eigen::Vector2d &observed_pt ) :
            cam_model(cam_model),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec),
            pattern_pt(pattern_pt),
            observed_pt( observed_pt) {}

    template <typename T>
    bool operator()(const double* const board2tcp,
                    const double* const h2e,
                    const double* const cam2cam,
                    T* residuals) const
    {
        T ptn_pt[3] = {T(pattern_pt(0)), T(pattern_pt(1)), T(pattern_pt(2))},
                tcp_pt[3], base_pt[3], cam_pt[3], cam2_pt[3];

        // Apply the current board2tcp transformation
        ceres::AngleAxisRotatePoint(board2tcp, ptn_pt, tcp_pt);

        tcp_pt[0] += board2tcp[3];
        tcp_pt[1] += board2tcp[4];
        tcp_pt[2] += board2tcp[5];


        T robot_r[3] = {T(robot_r_vec(0)), T(robot_r_vec(1)), T(robot_r_vec(2))},
                robot_t[3] = {T(robot_t_vec(0)), T(robot_t_vec(1)), T(robot_t_vec(2))};

        // Apply the robot position transformation
        ceres::AngleAxisRotatePoint(robot_r, tcp_pt, base_pt);

        base_pt[0] += robot_t[0];
        base_pt[1] += robot_t[1];
        base_pt[2] += robot_t[2];

        // Apply the current h2e transformation
        ceres::AngleAxisRotatePoint(h2e, base_pt, cam_pt);

        cam_pt[0] += h2e[3];
        cam_pt[1] += h2e[4];
        cam_pt[2] += h2e[5];


        // Apply the current cam2cam transformation
        ceres::AngleAxisRotatePoint(cam2cam, cam_pt, cam2_pt);

        cam2_pt[0] += cam2cam[3];
        cam2_pt[1] += cam2cam[4];
        cam2_pt[2] += cam2cam[5];

        // Projection.
        T proj_pt[2];
        cam_model.project(cam2_pt, proj_pt);

        // The error is the difference between the predicted and observed position.
        residuals[0] = proj_pt[0] - T(observed_pt(0));
        residuals[1] = proj_pt[1] - T(observed_pt(1));

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const PinholeCameraModel &cam_model,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec,
                                        const Eigen::Vector3d &pattern_pt,
                                        const Eigen::Vector2d &observed_pt )
    {
        return (new ceres::NumericDiffCostFunction<MultiCalibReprojectionError, ceres::CENTRAL, 2, 6, 6, 6>(
                new MultiCalibReprojectionError( cam_model, robot_r_vec, robot_t_vec, pattern_pt, observed_pt )));
    }

    const PinholeCameraModel &cam_model;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
    Eigen::Vector3d pattern_pt;
    Eigen::Vector2d observed_pt;
};



struct MultiEyeInHandCalibReprojectionError
{
    MultiEyeInHandCalibReprojectionError( const PinholeCameraModel &cam_model,
                                          const Eigen::Vector3d &robot_r_vec,
                                          const Eigen::Vector3d &robot_t_vec,
                                          const Eigen::Vector3d &pattern_pt,
                                          const Eigen::Vector2d &observed_pt ) :
            cam_model(cam_model),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec),
            pattern_pt(pattern_pt),
            observed_pt( observed_pt) {}

    bool operator()(const double* const b2w,
                    const double* const h2e,
                    const double* const c2c,
                    double* residuals) const
    {
        double ptn_pt[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))},
                tcp_pt[3], base_pt[3], cam_pt[3], cam2_pt[3], rot_mat[9], rot_mat_inv[9], rot_vec[3], tras_vec[3];

        // Apply the current board2tcp transformation
        ceres::AngleAxisRotatePoint(b2w, ptn_pt, base_pt);

        base_pt[0] += b2w[3];
        base_pt[1] += b2w[4];
        base_pt[2] += b2w[5];


        double robot_r[3] = {double(robot_r_vec(0)), double(robot_r_vec(1)), double(robot_r_vec(2))},
                robot_t[3] = {double(robot_t_vec(0)), double(robot_t_vec(1)), double(robot_t_vec(2))};

        // Apply the robot position transformation
        ceres::AngleAxisRotatePoint(robot_r, base_pt, tcp_pt);

        tcp_pt[0] += robot_t[0];
        tcp_pt[1] += robot_t[1];
        tcp_pt[2] += robot_t[2];





        //###########################################################################################

        ceres::AngleAxisToRotationMatrix(h2e, rot_mat);

        cv::Mat temp_g = (cv::Mat_<double>(4,4) << rot_mat[0], rot_mat[3], rot_mat[6], h2e[3],
                rot_mat[1], rot_mat[4], rot_mat[7], h2e[4],
                rot_mat[2], rot_mat[5], rot_mat[8], h2e[5],
                0, 0, 0, 1);

        cv::Mat temp_g_inv = temp_g.inv();

        rot_mat_inv[0] = temp_g_inv.at<double>(0,0);
        rot_mat_inv[1] = temp_g_inv.at<double>(1,0);
        rot_mat_inv[2] = temp_g_inv.at<double>(2,0);
        rot_mat_inv[3] = temp_g_inv.at<double>(0,1);
        rot_mat_inv[4] = temp_g_inv.at<double>(1,1);
        rot_mat_inv[5] = temp_g_inv.at<double>(2,1);
        rot_mat_inv[6] = temp_g_inv.at<double>(0,2);
        rot_mat_inv[7] = temp_g_inv.at<double>(1,2);
        rot_mat_inv[8] = temp_g_inv.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv, rot_vec);

        tras_vec[0] = temp_g_inv.at<double>(0,3);
        tras_vec[1] = temp_g_inv.at<double>(1,3);
        tras_vec[2] = temp_g_inv.at<double>(2,3);

        //###########################################################################################


        // Apply the current h2e transformation
        ceres::AngleAxisRotatePoint(rot_vec, tcp_pt, cam_pt);

        cam_pt[0] += tras_vec[0];
        cam_pt[1] += tras_vec[1];
        cam_pt[2] += tras_vec[2];


        // Apply the current cam2cam transformation
        ceres::AngleAxisRotatePoint(c2c, cam_pt, cam2_pt);

        cam2_pt[0] += c2c[3];
        cam2_pt[1] += c2c[4];
        cam2_pt[2] += c2c[5];

        // Projection.
        double proj_pt[2];
        cam_model.project(cam2_pt, proj_pt);

        // The error is the difference between the predicted and observed position.
        residuals[0] = proj_pt[0] - double(observed_pt(0));
        residuals[1] = proj_pt[1] - double(observed_pt(1));

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const PinholeCameraModel &cam_model,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec,
                                        const Eigen::Vector3d &pattern_pt,
                                        const Eigen::Vector2d &observed_pt )
    {
        return (new ceres::NumericDiffCostFunction<MultiEyeInHandCalibReprojectionError, ceres::CENTRAL, 2, 6, 6, 6>(
                new MultiEyeInHandCalibReprojectionError( cam_model, robot_r_vec, robot_t_vec, pattern_pt, observed_pt )));
    }

    const PinholeCameraModel &cam_model;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
    Eigen::Vector3d pattern_pt;
    Eigen::Vector2d observed_pt;
};


struct MobileOptEyeInHandCalibReprojectionError
{
    MobileOptEyeInHandCalibReprojectionError( const PinholeCameraModel &cam_model,
                                          const Eigen::Vector3d &robot_r_vec,
                                          const Eigen::Vector3d &robot_t_vec,
                                          const Eigen::Vector3d &pattern_pt,
                                          const Eigen::Vector2d &observed_pt ) :
            cam_model(cam_model),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec),
            pattern_pt(pattern_pt),
            observed_pt( observed_pt) {}

    //template <typename T>
    bool operator()(const double* const b2w,
                    const double* const h2e,
                    const double* const c2c,
                    const double* const robot_opt,
                    double* residuals) const
    {
        double ptn_pt[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))},
                tcp_pt[3], tcp_pt_opt[3], base_pt[3], cam_pt[3], cam2_pt[3];

        // Apply the current board2tcp transformation
        ceres::AngleAxisRotatePoint(b2w, ptn_pt, base_pt);

        base_pt[0] += b2w[3];
        base_pt[1] += b2w[4];
        base_pt[2] += b2w[5];


        double robot_r[3] = {double(robot_r_vec(0)), double(robot_r_vec(1)), double(robot_r_vec(2))},
                robot_t[3] = {double(robot_t_vec(0)), double(robot_t_vec(1)), double(robot_t_vec(2))};

        // Apply the robot position transformation
        ceres::AngleAxisRotatePoint(robot_r, base_pt, tcp_pt);

        tcp_pt[0] += robot_t[0];
        tcp_pt[1] += robot_t[1];
        tcp_pt[2] += robot_t[2];

        // Apply the robot position transformation
        ceres::AngleAxisRotatePoint(robot_opt, tcp_pt, tcp_pt_opt);

        tcp_pt_opt[0] += robot_opt[3];
        tcp_pt_opt[1] += robot_opt[4];
        tcp_pt_opt[2] += robot_opt[5];

        // Apply the current h2e transformation
        ceres::AngleAxisRotatePoint(h2e, tcp_pt_opt, cam_pt);

        cam_pt[0] += h2e[3];
        cam_pt[1] += h2e[4];
        cam_pt[2] += h2e[5];


        // Apply the current cam2cam transformation
        ceres::AngleAxisRotatePoint(c2c, cam_pt, cam2_pt);

        cam2_pt[0] += c2c[3];
        cam2_pt[1] += c2c[4];
        cam2_pt[2] += c2c[5];

        // Projection.
        double proj_pt[2];
        cam_model.project(cam2_pt, proj_pt);

        // The error is the difference between the predicted and observed position.
        residuals[0] = proj_pt[0] - double(observed_pt(0));
        residuals[1] = proj_pt[1] - double(observed_pt(1));

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const PinholeCameraModel &cam_model,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec,
                                        const Eigen::Vector3d &pattern_pt,
                                        const Eigen::Vector2d &observed_pt )
    {
        return (new ceres::NumericDiffCostFunction<MobileOptEyeInHandCalibReprojectionError, ceres::CENTRAL, 2, 6, 6, 6, 6>(
                new MobileOptEyeInHandCalibReprojectionError( cam_model, robot_r_vec, robot_t_vec, pattern_pt, observed_pt )));
    }

    const PinholeCameraModel &cam_model;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
    Eigen::Vector3d pattern_pt;
    Eigen::Vector2d observed_pt;
};



struct MobileOptEyeToHandCalibReprojectionError
{
    MobileOptEyeToHandCalibReprojectionError( const PinholeCameraModel &cam_model,
                                              const Eigen::Vector3d &robot_r_vec,
                                              const Eigen::Vector3d &robot_t_vec,
                                              const Eigen::Vector3d &pattern_pt,
                                              const Eigen::Vector2d &observed_pt ) :
            cam_model(cam_model),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec),
            pattern_pt(pattern_pt),
            observed_pt( observed_pt) {}

    template <typename T>
    bool operator()(const T* const b2w,
                    const T* const h2e,
                    const T* const c2c,
                    const T* const robot_opt,
                    T* residuals) const
    {
        T ptn_pt[3] = {T(pattern_pt(0)), T(pattern_pt(1)), T(pattern_pt(2))},
                tcp_pt[3], tcp_pt_opt[3], base_pt[3], cam_pt[3], cam2_pt[3];

        // Apply the current board2tcp transformation
        ceres::AngleAxisRotatePoint(b2w, ptn_pt, tcp_pt_opt);

        tcp_pt_opt[0] += b2w[3];
        tcp_pt_opt[1] += b2w[4];
        tcp_pt_opt[2] += b2w[5];

        // Apply the robot position transformation
        ceres::AngleAxisRotatePoint(robot_opt, tcp_pt_opt, tcp_pt);

        tcp_pt[0] += robot_opt[3];
        tcp_pt[1] += robot_opt[4];
        tcp_pt[2] += robot_opt[5];

        T robot_r[3] = {T(robot_r_vec(0)), T(robot_r_vec(1)), T(robot_r_vec(2))},
                robot_t[3] = {T(robot_t_vec(0)), T(robot_t_vec(1)), T(robot_t_vec(2))};

        // Apply the robot position transformation
        ceres::AngleAxisRotatePoint(robot_r, tcp_pt, base_pt);

        base_pt[0] += robot_t[0];
        base_pt[1] += robot_t[1];
        base_pt[2] += robot_t[2];

        // Apply the current h2e transformation
        ceres::AngleAxisRotatePoint(h2e, base_pt, cam_pt);

        cam_pt[0] += h2e[3];
        cam_pt[1] += h2e[4];
        cam_pt[2] += h2e[5];


        // Apply the current cam2cam transformation
        ceres::AngleAxisRotatePoint(c2c, cam_pt, cam2_pt);

        cam2_pt[0] += c2c[3];
        cam2_pt[1] += c2c[4];
        cam2_pt[2] += c2c[5];

        // Projection.
        T proj_pt[2];
        cam_model.project(cam2_pt, proj_pt);

        // The error is the difference between the predicted and observed position.
        residuals[0] = proj_pt[0] - T(observed_pt(0));
        residuals[1] = proj_pt[1] - T(observed_pt(1));

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const PinholeCameraModel &cam_model,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec,
                                        const Eigen::Vector3d &pattern_pt,
                                        const Eigen::Vector2d &observed_pt )
    {
        return (new ceres::AutoDiffCostFunction<MobileOptEyeToHandCalibReprojectionError, 2, 6, 6, 6, 6>(
                new MobileOptEyeToHandCalibReprojectionError( cam_model, robot_r_vec, robot_t_vec, pattern_pt, observed_pt )));
    }

    const PinholeCameraModel &cam_model;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
    Eigen::Vector3d pattern_pt;
    Eigen::Vector2d observed_pt;
};



struct MultiMobileCalibReprojectionError
{
    MultiMobileCalibReprojectionError( const PinholeCameraModel &cam_model,
                                          const Eigen::Vector3d &pattern_pt,
                                          const Eigen::Vector2d &observed_pt ) :
            cam_model(cam_model),
            pattern_pt(pattern_pt),
            observed_pt( observed_pt) {}

    bool operator()(const double* const b2w,
                    const double* const h2e,
                    const double* const c2c,
                    const double* const robot_opt,
                    double* residuals) const
    {
        double ptn_pt[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))},
                tcp_pt[3], base_pt[3], cam_pt[3], cam2_pt[3];

        // Apply the current board2tcp transformation
        ceres::AngleAxisRotatePoint(b2w, ptn_pt, base_pt);

        base_pt[0] += b2w[3];
        base_pt[1] += b2w[4];
        base_pt[2] += b2w[5];


        double robot_r[3] = {double(robot_opt[0]), double(robot_opt[1]), double(robot_opt[2])},
                robot_t[3] = {double(robot_opt[3]), double(robot_opt[4]), double(robot_opt[5])};

        // Apply the robot position transformation
        ceres::AngleAxisRotatePoint(robot_r, base_pt, tcp_pt);

        tcp_pt[0] += robot_t[0];
        tcp_pt[1] += robot_t[1];
        tcp_pt[2] += robot_t[2];

        // Apply the current h2e transformation
        ceres::AngleAxisRotatePoint(h2e, tcp_pt, cam_pt);

        cam_pt[0] += h2e[3];
        cam_pt[1] += h2e[4];
        cam_pt[2] += h2e[5];


        // Apply the current cam2cam transformation
        ceres::AngleAxisRotatePoint(c2c, cam_pt, cam2_pt);

        cam2_pt[0] += c2c[3];
        cam2_pt[1] += c2c[4];
        cam2_pt[2] += c2c[5];

        // Projection.
        double proj_pt[2];
        cam_model.project(cam2_pt, proj_pt);

        // The error is the difference between the predicted and observed position.
        residuals[0] = proj_pt[0] - double(observed_pt(0));
        residuals[1] = proj_pt[1] - double(observed_pt(1));

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const PinholeCameraModel &cam_model,
                                        const Eigen::Vector3d &pattern_pt,
                                        const Eigen::Vector2d &observed_pt )
    {
        return (new ceres::NumericDiffCostFunction<MultiMobileCalibReprojectionError, ceres::CENTRAL, 2, 6, 6, 6, 6>(
                new MultiMobileCalibReprojectionError( cam_model, pattern_pt, observed_pt )));
    }

    const PinholeCameraModel &cam_model;
    Eigen::Vector3d pattern_pt;
    Eigen::Vector2d observed_pt;
};

// Class for multi camera inverse
struct MultiCalibReprojectionErrorInverse
{
    MultiCalibReprojectionErrorInverse( const PinholeCameraModel &cam_model,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec,
                                        const Eigen::Vector3d &pattern_pt,
                                        const Eigen::Vector2d &observed_pt ) :
            cam_model(cam_model),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec),
            pattern_pt(pattern_pt),
            observed_pt( observed_pt) {}

    //template <typename T>
    bool operator()(const double* const board2tcp,
                    const double* const h2e,
                    const double* const cam2cam,
                    double* residuals) const
    {
        double ptn_pt[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))},
                tcp_pt[3], base_pt[3], cam_pt[3], cam2_pt[3], tras_vec[3], rot_vec[3], rot_mat[9], rot_mat_inv[9];

        // Apply the current board2tcp transformation
        ceres::AngleAxisRotatePoint(board2tcp, ptn_pt, tcp_pt);

        tcp_pt[0] += board2tcp[3];
        tcp_pt[1] += board2tcp[4];
        tcp_pt[2] += board2tcp[5];


        double robot_r[3] = {double(robot_r_vec(0)), double(robot_r_vec(1)), double(robot_r_vec(2))},
                robot_t[3] = {double(robot_t_vec(0)), double(robot_t_vec(1)), double(robot_t_vec(2))};

        // Apply the robot position transformation
        ceres::AngleAxisRotatePoint(robot_r, tcp_pt, base_pt);

        base_pt[0] += robot_t[0];
        base_pt[1] += robot_t[1];
        base_pt[2] += robot_t[2];

        // Apply the current h2e transformation
        ceres::AngleAxisRotatePoint(h2e, base_pt, cam_pt);

        cam_pt[0] += h2e[3];
        cam_pt[1] += h2e[4];
        cam_pt[2] += h2e[5];

        //###########################################################################################
        cv::Mat temp_variable;

        cv::Mat cam2cam_mat_inv;
        cv::Mat cam2cam_mat = (cv::Mat_<double>(1,3) << cam2cam[0], cam2cam[1], cam2cam[2]);
        ceres::AngleAxisToRotationMatrix(cam2cam, rot_mat);

        cv::Mat temp_g = (cv::Mat_<double>(4,4) <<
                                                rot_mat[0], rot_mat[3], rot_mat[6], cam2cam[3],
                rot_mat[1], rot_mat[4], rot_mat[7], cam2cam[4],
                rot_mat[2], rot_mat[5], rot_mat[8], cam2cam[5],
                0, 0, 0, 1);

        cv::Mat temp_g_inv = temp_g.inv();

        rot_mat_inv[0] = temp_g_inv.at<double>(0,0);
        rot_mat_inv[1] = temp_g_inv.at<double>(1,0);
        rot_mat_inv[2] = temp_g_inv.at<double>(2,0);
        rot_mat_inv[3] = temp_g_inv.at<double>(0,1);
        rot_mat_inv[4] = temp_g_inv.at<double>(1,1);
        rot_mat_inv[5] = temp_g_inv.at<double>(2,1);
        rot_mat_inv[6] = temp_g_inv.at<double>(0,2);
        rot_mat_inv[7] = temp_g_inv.at<double>(1,2);
        rot_mat_inv[8] = temp_g_inv.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv, rot_vec);

        tras_vec[0] = temp_g_inv.at<double>(0,3);
        tras_vec[1] = temp_g_inv.at<double>(1,3);
        tras_vec[2] = temp_g_inv.at<double>(2,3);

        //###########################################################################################

        ceres::AngleAxisRotatePoint(rot_vec, cam_pt, cam2_pt);

        cam2_pt[0] += tras_vec[0];
        cam2_pt[1] += tras_vec[1];
        cam2_pt[2] += tras_vec[2];

        // Projection.
        double proj_pt[2];
        cam_model.project(cam2_pt, proj_pt);

        // The error is the difference between the predicted and observed position.
        residuals[0] = proj_pt[0] - double(observed_pt(0));
        residuals[1] = proj_pt[1] - double(observed_pt(1));

        return true;
    }

    static ceres::CostFunction* Create( const PinholeCameraModel &cam_model,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec,
                                        const Eigen::Vector3d &pattern_pt,
                                        const Eigen::Vector2d &observed_pt )
    {
        return (new ceres::NumericDiffCostFunction<MultiCalibReprojectionErrorInverse, ceres::CENTRAL, 2, 6, 6, 6>(
                new MultiCalibReprojectionErrorInverse( cam_model, robot_r_vec, robot_t_vec, pattern_pt, observed_pt )));
    }

    const PinholeCameraModel &cam_model;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
    Eigen::Vector3d pattern_pt;
    Eigen::Vector2d observed_pt;
};


// Class for multi camera inverse
struct MultiEyeInHandCalibReprojectionErrorInverse
{
    MultiEyeInHandCalibReprojectionErrorInverse( const PinholeCameraModel &cam_model,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec,
                                        const Eigen::Vector3d &pattern_pt,
                                        const Eigen::Vector2d &observed_pt ) :
            cam_model(cam_model),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec),
            pattern_pt(pattern_pt),
            observed_pt( observed_pt) {}

    //template <typename T>
    bool operator()(const double* const b2w,
                    const double* const h2e,
                    const double* const c2c,
                    double* residuals) const
    {
        double ptn_pt[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))},
                tcp_pt[3], base_pt[3], cam_pt[3], cam2_pt[3], tras_vec[3], rot_vec[3], rot_mat[9], rot_mat_inv[9];

        // Apply the current board2tcp transformation
        ceres::AngleAxisRotatePoint(b2w, ptn_pt, base_pt);

        base_pt[0] += b2w[3];
        base_pt[1] += b2w[4];
        base_pt[2] += b2w[5];


        double robot_r[3] = {double(robot_r_vec(0)), double(robot_r_vec(1)), double(robot_r_vec(2))},
                robot_t[3] = {double(robot_t_vec(0)), double(robot_t_vec(1)), double(robot_t_vec(2))};

        // Apply the robot position transformation
        ceres::AngleAxisRotatePoint(robot_r, base_pt, tcp_pt);

        tcp_pt[0] += robot_t[0];
        tcp_pt[1] += robot_t[1];
        tcp_pt[2] += robot_t[2];

        // Apply the current h2e transformation
        ceres::AngleAxisRotatePoint(h2e, tcp_pt, cam_pt);

        cam_pt[0] += h2e[3];
        cam_pt[1] += h2e[4];
        cam_pt[2] += h2e[5];

        //###########################################################################################
        cv::Mat temp_variable;

        cv::Mat cam2cam_mat_inv;
        cv::Mat cam2cam_mat = (cv::Mat_<double>(1,3) << c2c[0], c2c[1], c2c[2]);
        ceres::AngleAxisToRotationMatrix(c2c, rot_mat);

        cv::Mat temp_g = (cv::Mat_<double>(4,4) <<
                rot_mat[0], rot_mat[3], rot_mat[6], c2c[3],
                rot_mat[1], rot_mat[4], rot_mat[7], c2c[4],
                rot_mat[2], rot_mat[5], rot_mat[8], c2c[5],
                0, 0, 0, 1);

        cv::Mat temp_g_inv = temp_g.inv();

        rot_mat_inv[0] = temp_g_inv.at<double>(0,0);
        rot_mat_inv[1] = temp_g_inv.at<double>(1,0);
        rot_mat_inv[2] = temp_g_inv.at<double>(2,0);
        rot_mat_inv[3] = temp_g_inv.at<double>(0,1);
        rot_mat_inv[4] = temp_g_inv.at<double>(1,1);
        rot_mat_inv[5] = temp_g_inv.at<double>(2,1);
        rot_mat_inv[6] = temp_g_inv.at<double>(0,2);
        rot_mat_inv[7] = temp_g_inv.at<double>(1,2);
        rot_mat_inv[8] = temp_g_inv.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv, rot_vec);

        tras_vec[0] = temp_g_inv.at<double>(0,3);
        tras_vec[1] = temp_g_inv.at<double>(1,3);
        tras_vec[2] = temp_g_inv.at<double>(2,3);

        //###########################################################################################

        ceres::AngleAxisRotatePoint(rot_vec, cam_pt, cam2_pt);

        cam2_pt[0] += tras_vec[0];
        cam2_pt[1] += tras_vec[1];
        cam2_pt[2] += tras_vec[2];

        // Projection.
        double proj_pt[2];
        cam_model.project(cam2_pt, proj_pt);

        // The error is the difference between the predicted and observed position.
        residuals[0] = proj_pt[0] - double(observed_pt(0));
        residuals[1] = proj_pt[1] - double(observed_pt(1));

        return true;
    }

    static ceres::CostFunction* Create( const PinholeCameraModel &cam_model,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec,
                                        const Eigen::Vector3d &pattern_pt,
                                        const Eigen::Vector2d &observed_pt )
    {
        return (new ceres::NumericDiffCostFunction<MultiEyeInHandCalibReprojectionErrorInverse, ceres::CENTRAL, 2, 6, 6, 6>(
                new MultiEyeInHandCalibReprojectionErrorInverse( cam_model, robot_r_vec, robot_t_vec, pattern_pt, observed_pt )));
    }

    const PinholeCameraModel &cam_model;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
    Eigen::Vector3d pattern_pt;
    Eigen::Vector2d observed_pt;
};


struct MobileOptEyeInHandCalibReprojectionErrorInverse
{
    MobileOptEyeInHandCalibReprojectionErrorInverse( const PinholeCameraModel &cam_model,
                                                 const Eigen::Vector3d &robot_r_vec,
                                                 const Eigen::Vector3d &robot_t_vec,
                                                 const Eigen::Vector3d &pattern_pt,
                                                 const Eigen::Vector2d &observed_pt ) :
            cam_model(cam_model),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec),
            pattern_pt(pattern_pt),
            observed_pt( observed_pt) {}

    //template <typename T>
    bool operator()(const double* const b2w,
                    const double* const h2e,
                    const double* const c2c,
                    const double* const robot_opt,
                    double* residuals) const
    {
        double ptn_pt[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))},
                tcp_pt[3], tcp_pt_opt[3], base_pt[3], cam_pt[3], cam2_pt[3], tras_vec[3], rot_vec[3], rot_mat[9], rot_mat_inv[9];

        // Apply the current board2tcp transformation
        ceres::AngleAxisRotatePoint(b2w, ptn_pt, base_pt);

        base_pt[0] += b2w[3];
        base_pt[1] += b2w[4];
        base_pt[2] += b2w[5];


        double robot_r[3] = {double(robot_r_vec(0)), double(robot_r_vec(1)), double(robot_r_vec(2))},
                robot_t[3] = {double(robot_t_vec(0)), double(robot_t_vec(1)), double(robot_t_vec(2))};

        // Apply the robot position transformation
        ceres::AngleAxisRotatePoint(robot_r, base_pt, tcp_pt);

        tcp_pt[0] += robot_t[0];
        tcp_pt[1] += robot_t[1];
        tcp_pt[2] += robot_t[2];

        // Apply the robot position transformation
        ceres::AngleAxisRotatePoint(robot_opt, tcp_pt, tcp_pt_opt);

        tcp_pt_opt[0] += robot_opt[3];
        tcp_pt_opt[1] += robot_opt[4];
        tcp_pt_opt[2] += robot_opt[5];

        // Apply the current h2e transformation
        ceres::AngleAxisRotatePoint(h2e, tcp_pt_opt, cam_pt);

        cam_pt[0] += h2e[3];
        cam_pt[1] += h2e[4];
        cam_pt[2] += h2e[5];

        //###########################################################################################
        cv::Mat temp_variable;

        cv::Mat cam2cam_mat_inv;
        cv::Mat cam2cam_mat = (cv::Mat_<double>(1,3) << c2c[0], c2c[1], c2c[2]);
        ceres::AngleAxisToRotationMatrix(c2c, rot_mat);

        cv::Mat temp_g = (cv::Mat_<double>(4,4) <<
                                                rot_mat[0], rot_mat[3], rot_mat[6], c2c[3],
                rot_mat[1], rot_mat[4], rot_mat[7], c2c[4],
                rot_mat[2], rot_mat[5], rot_mat[8], c2c[5],
                0, 0, 0, 1);

        cv::Mat temp_g_inv = temp_g.inv();

        rot_mat_inv[0] = temp_g_inv.at<double>(0,0);
        rot_mat_inv[1] = temp_g_inv.at<double>(1,0);
        rot_mat_inv[2] = temp_g_inv.at<double>(2,0);
        rot_mat_inv[3] = temp_g_inv.at<double>(0,1);
        rot_mat_inv[4] = temp_g_inv.at<double>(1,1);
        rot_mat_inv[5] = temp_g_inv.at<double>(2,1);
        rot_mat_inv[6] = temp_g_inv.at<double>(0,2);
        rot_mat_inv[7] = temp_g_inv.at<double>(1,2);
        rot_mat_inv[8] = temp_g_inv.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv, rot_vec);

        tras_vec[0] = temp_g_inv.at<double>(0,3);
        tras_vec[1] = temp_g_inv.at<double>(1,3);
        tras_vec[2] = temp_g_inv.at<double>(2,3);

        //###########################################################################################

        ceres::AngleAxisRotatePoint(rot_vec, cam_pt, cam2_pt);

        cam2_pt[0] += tras_vec[0];
        cam2_pt[1] += tras_vec[1];
        cam2_pt[2] += tras_vec[2];

        // Projection.
        double proj_pt[2];
        cam_model.project(cam2_pt, proj_pt);

        // The error is the difference between the predicted and observed position.
        residuals[0] = proj_pt[0] - double(observed_pt(0));
        residuals[1] = proj_pt[1] - double(observed_pt(1));

        return true;
    }

    static ceres::CostFunction* Create( const PinholeCameraModel &cam_model,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec,
                                        const Eigen::Vector3d &pattern_pt,
                                        const Eigen::Vector2d &observed_pt )
    {
        return (new ceres::NumericDiffCostFunction<MobileOptEyeInHandCalibReprojectionErrorInverse, ceres::CENTRAL, 2, 6, 6, 6, 6>(
                new MobileOptEyeInHandCalibReprojectionErrorInverse( cam_model, robot_r_vec, robot_t_vec, pattern_pt, observed_pt )));
    }

    const PinholeCameraModel &cam_model;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
    Eigen::Vector3d pattern_pt;
    Eigen::Vector2d observed_pt;
};


struct MobileOptEyeToHandCalibReprojectionErrorInverse
{
    MobileOptEyeToHandCalibReprojectionErrorInverse( const PinholeCameraModel &cam_model,
                                                     const Eigen::Vector3d &robot_r_vec,
                                                     const Eigen::Vector3d &robot_t_vec,
                                                     const Eigen::Vector3d &pattern_pt,
                                                     const Eigen::Vector2d &observed_pt ) :
            cam_model(cam_model),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec),
            pattern_pt(pattern_pt),
            observed_pt( observed_pt) {}

    //template <typename T>
    bool operator()(const double* const b2w,
                    const double* const h2e,
                    const double* const c2c,
                    const double* const robot_opt,
                    double* residuals) const
    {
        double ptn_pt[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))},
                tcp_pt[3], tcp_pt_opt[3], base_pt[3], cam_pt[3], cam2_pt[3], tras_vec[3], rot_vec[3], rot_mat[9], rot_mat_inv[9];

        // Apply the current board2tcp transformation
        ceres::AngleAxisRotatePoint(b2w, ptn_pt, tcp_pt_opt);

        tcp_pt_opt[0] += b2w[3];
        tcp_pt_opt[1] += b2w[4];
        tcp_pt_opt[2] += b2w[5];

        // Apply the robot position transformation
        ceres::AngleAxisRotatePoint(robot_opt, tcp_pt_opt, tcp_pt);

        tcp_pt[0] += robot_opt[3];
        tcp_pt[1] += robot_opt[4];
        tcp_pt[2] += robot_opt[5];

        double robot_r[3] = {double(robot_r_vec(0)), double(robot_r_vec(1)), double(robot_r_vec(2))},
                robot_t[3] = {double(robot_t_vec(0)), double(robot_t_vec(1)), double(robot_t_vec(2))};

        // Apply the robot position transformation
        ceres::AngleAxisRotatePoint(robot_r, tcp_pt, base_pt);

        base_pt[0] += robot_t[0];
        base_pt[1] += robot_t[1];
        base_pt[2] += robot_t[2];

        // Apply the current h2e transformation
        ceres::AngleAxisRotatePoint(h2e, base_pt, cam_pt);

        cam_pt[0] += h2e[3];
        cam_pt[1] += h2e[4];
        cam_pt[2] += h2e[5];

        //###########################################################################################
        ceres::AngleAxisToRotationMatrix(c2c, rot_mat);

        cv::Mat temp_g = (cv::Mat_<double>(4,4) <<
                rot_mat[0], rot_mat[3], rot_mat[6], c2c[3],
                rot_mat[1], rot_mat[4], rot_mat[7], c2c[4],
                rot_mat[2], rot_mat[5], rot_mat[8], c2c[5],
                0, 0, 0, 1);

        cv::Mat temp_g_inv = temp_g.inv();

        rot_mat_inv[0] = temp_g_inv.at<double>(0,0);
        rot_mat_inv[1] = temp_g_inv.at<double>(1,0);
        rot_mat_inv[2] = temp_g_inv.at<double>(2,0);
        rot_mat_inv[3] = temp_g_inv.at<double>(0,1);
        rot_mat_inv[4] = temp_g_inv.at<double>(1,1);
        rot_mat_inv[5] = temp_g_inv.at<double>(2,1);
        rot_mat_inv[6] = temp_g_inv.at<double>(0,2);
        rot_mat_inv[7] = temp_g_inv.at<double>(1,2);
        rot_mat_inv[8] = temp_g_inv.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv, rot_vec);

        tras_vec[0] = temp_g_inv.at<double>(0,3);
        tras_vec[1] = temp_g_inv.at<double>(1,3);
        tras_vec[2] = temp_g_inv.at<double>(2,3);

        //###########################################################################################

        ceres::AngleAxisRotatePoint(rot_vec, cam_pt, cam2_pt);

        cam2_pt[0] += tras_vec[0];
        cam2_pt[1] += tras_vec[1];
        cam2_pt[2] += tras_vec[2];

        // Projection.
        double proj_pt[2];
        cam_model.project(cam2_pt, proj_pt);

        // The error is the difference between the predicted and observed position.
        residuals[0] = proj_pt[0] - double(observed_pt(0));
        residuals[1] = proj_pt[1] - double(observed_pt(1));

        return true;
    }

    static ceres::CostFunction* Create( const PinholeCameraModel &cam_model,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec,
                                        const Eigen::Vector3d &pattern_pt,
                                        const Eigen::Vector2d &observed_pt )
    {
        return (new ceres::NumericDiffCostFunction<MobileOptEyeToHandCalibReprojectionErrorInverse, ceres::CENTRAL, 2, 6, 6, 6, 6>(
                new MobileOptEyeToHandCalibReprojectionErrorInverse( cam_model, robot_r_vec, robot_t_vec, pattern_pt, observed_pt )));
    }

    const PinholeCameraModel &cam_model;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
    Eigen::Vector3d pattern_pt;
    Eigen::Vector2d observed_pt;
};


struct MultiMobileCalibReprojectionErrorInverse
{
    MultiMobileCalibReprojectionErrorInverse( const PinholeCameraModel &cam_model,
                                                 const Eigen::Vector3d &pattern_pt,
                                                 const Eigen::Vector2d &observed_pt ) :
            cam_model(cam_model),
            pattern_pt(pattern_pt),
            observed_pt( observed_pt) {}

    //template <typename T>
    bool operator()(const double* const b2w,
                    const double* const h2e,
                    const double* const c2c,
                    const double* const robot_opt,
                    double* residuals) const
    {
        double ptn_pt[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))},
                tcp_pt[3], base_pt[3], cam_pt[3], cam2_pt[3], tras_vec[3], rot_vec[3], rot_mat[9], rot_mat_inv[9];

        // Apply the current board2tcp transformation
        ceres::AngleAxisRotatePoint(b2w, ptn_pt, base_pt);

        base_pt[0] += b2w[3];
        base_pt[1] += b2w[4];
        base_pt[2] += b2w[5];


        double robot_r[3] = {double(robot_opt[0]), double(robot_opt[1]), double(robot_opt[2])},
                robot_t[3] = {double(robot_opt[3]), double(robot_opt[4]), double(robot_opt[5])};

        // Apply the robot position transformation
        ceres::AngleAxisRotatePoint(robot_r, base_pt, tcp_pt);

        tcp_pt[0] += robot_t[0];
        tcp_pt[1] += robot_t[1];
        tcp_pt[2] += robot_t[2];

        // Apply the current h2e transformation
        ceres::AngleAxisRotatePoint(h2e, tcp_pt, cam_pt);

        cam_pt[0] += h2e[3];
        cam_pt[1] += h2e[4];
        cam_pt[2] += h2e[5];

        //###########################################################################################
        cv::Mat temp_variable;

        cv::Mat cam2cam_mat_inv;
        cv::Mat cam2cam_mat = (cv::Mat_<double>(1,3) << c2c[0], c2c[1], c2c[2]);
        ceres::AngleAxisToRotationMatrix(c2c, rot_mat);

        cv::Mat temp_g = (cv::Mat_<double>(4,4) <<
                                                rot_mat[0], rot_mat[3], rot_mat[6], c2c[3],
                rot_mat[1], rot_mat[4], rot_mat[7], c2c[4],
                rot_mat[2], rot_mat[5], rot_mat[8], c2c[5],
                0, 0, 0, 1);

        cv::Mat temp_g_inv = temp_g.inv();

        rot_mat_inv[0] = temp_g_inv.at<double>(0,0);
        rot_mat_inv[1] = temp_g_inv.at<double>(1,0);
        rot_mat_inv[2] = temp_g_inv.at<double>(2,0);
        rot_mat_inv[3] = temp_g_inv.at<double>(0,1);
        rot_mat_inv[4] = temp_g_inv.at<double>(1,1);
        rot_mat_inv[5] = temp_g_inv.at<double>(2,1);
        rot_mat_inv[6] = temp_g_inv.at<double>(0,2);
        rot_mat_inv[7] = temp_g_inv.at<double>(1,2);
        rot_mat_inv[8] = temp_g_inv.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv, rot_vec);

        tras_vec[0] = temp_g_inv.at<double>(0,3);
        tras_vec[1] = temp_g_inv.at<double>(1,3);
        tras_vec[2] = temp_g_inv.at<double>(2,3);

        //###########################################################################################

        ceres::AngleAxisRotatePoint(rot_vec, cam_pt, cam2_pt);

        cam2_pt[0] += tras_vec[0];
        cam2_pt[1] += tras_vec[1];
        cam2_pt[2] += tras_vec[2];

        // Projection.
        double proj_pt[2];
        cam_model.project(cam2_pt, proj_pt);

        // The error is the difference between the predicted and observed position.
        residuals[0] = proj_pt[0] - double(observed_pt(0));
        residuals[1] = proj_pt[1] - double(observed_pt(1));

        return true;
    }

    static ceres::CostFunction* Create( const PinholeCameraModel &cam_model,
                                        const Eigen::Vector3d &pattern_pt,
                                        const Eigen::Vector2d &observed_pt )
    {
        return (new ceres::NumericDiffCostFunction<MultiMobileCalibReprojectionErrorInverse, ceres::CENTRAL, 2, 6, 6, 6, 6>(
                new MultiMobileCalibReprojectionErrorInverse( cam_model, pattern_pt, observed_pt )));
    }

    const PinholeCameraModel &cam_model;
    Eigen::Vector3d pattern_pt;
    Eigen::Vector2d observed_pt;
};



struct MobileCalibPnp
{
    MobileCalibPnp(const Eigen::Vector3d &pnp_r_vec,
                    const Eigen::Vector3d &pnp_t_vec,
                    const Eigen::Vector3d &pattern_pt) :
            pnp_r_vec(pnp_r_vec),
            pnp_t_vec(pnp_t_vec),
            pattern_pt(pattern_pt) {}

    //template <typename T>
    bool operator()(//const double* const weights,
            const double* const b2w,
            const double* const h2e,
            const double* const robot_opt,
            double* residuals) const
    {
        //T ptn_pt[3] = {T(pattern_pt(0)), T(pattern_pt(1)), T(pattern_pt(2))},
        //  tcp_pt[3], base_r[3], cam_pt[3];
        double ptn_pt[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))},
                tcp_pt[3], base_r[3], cam_pt[3], cam_pt_pnp[3];
        double ptn_pt2[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))};

        // Apply the current board2ee transformation
        ceres::AngleAxisRotatePoint(b2w, ptn_pt, base_r);

        base_r[0] += b2w[3];
        base_r[1] += b2w[4];
        base_r[2] += b2w[5];

        double robot_r[3] = {double(robot_opt[0]), double(robot_opt[1]), double(robot_opt[2])},
                robot_t[3] = {double(robot_opt[3]), double(robot_opt[4]), double(robot_opt[5])};

        // Apply the robot position transformation
        ceres::AngleAxisRotatePoint(robot_r, base_r, tcp_pt);

        tcp_pt[0] += robot_t[0];
        tcp_pt[1] += robot_t[1];
        tcp_pt[2] += robot_t[2];

        // Apply the current h2e transformation
        ceres::AngleAxisRotatePoint(h2e, tcp_pt, cam_pt);

        cam_pt[0] += h2e[3];
        cam_pt[1] += h2e[4];
        cam_pt[2] += h2e[5];

        //Chain2
        // Apply intrinsic parameters
        double pnp_r[3] = {double(pnp_r_vec(0)), double(pnp_r_vec(1)), double(pnp_r_vec(2))};
        double pnp_t[3] = {double(pnp_t_vec(0)), double(pnp_t_vec(1)), double(pnp_t_vec(2))};

        // Apply the current pnp transformation
        ceres::AngleAxisRotatePoint(pnp_r, ptn_pt2, cam_pt_pnp);

        cam_pt_pnp[0] += pnp_t[0];
        cam_pt_pnp[1] += pnp_t[1];
        cam_pt_pnp[2] += pnp_t[2];

        residuals[0] = (cam_pt[0] - cam_pt_pnp[0]);
        residuals[1] = (cam_pt[1] - cam_pt_pnp[1]);
        residuals[2] = (cam_pt[2] - cam_pt_pnp[2]);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const Eigen::Vector3d &pnp_r_vec,
                                        const Eigen::Vector3d &pnp_t_vec,
                                        const Eigen::Vector3d &pattern_pt)
    {
        return new ceres::NumericDiffCostFunction<MobileCalibPnp, ceres::CENTRAL, 3, 6, 6, 6>(
                new MobileCalibPnp(pnp_r_vec, pnp_t_vec, pattern_pt));
    }

    Eigen::Vector3d pnp_r_vec;
    Eigen::Vector3d pnp_t_vec;
    Eigen::Vector3d pattern_pt;
};


template <typename T>
void MatrixMultiply(const T* X, const T* Y, T* M, int row0, int col0, int row1, int col1){

    int xi, yi;
    T r;
    for (int i = 0; i < row0; i++){
        for (int j = 0; j < col1; j++){
            // dot product the ith row of X by the jth column of Y, results in the
            //cout << "i, j " << i << ", " << j << endl;

            r= T(0);
            for (int index = 0; index < col0; index++){
                xi = i*col0 + index; // walk across the row
                yi = index*col1 + j; // walk down the columm

                r += X[xi]*Y[yi];

                //cout << "Mult " << xi << " from first by " << yi << " from second" << endl;
            }
            //cout << "Result stored in " << i*col1 + j << endl;
            //char ch; cin >> ch;
            M[i*col1 + j] = r;
        }
    }
}


struct Classic_pnp
{
    Classic_pnp(const Eigen::Vector3d &pnp_r_vec,
                const Eigen::Vector3d &pnp_t_vec,
                const Eigen::Vector3d &robot_r_vec,
                const Eigen::Vector3d &robot_t_vec,
                   const Eigen::Vector3d &pattern_pt) :
            pnp_r_vec(pnp_r_vec),
            pnp_t_vec(pnp_t_vec),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec),
            pattern_pt(pattern_pt) {}

    template <typename T>
    bool operator()(const T* const b2w,
            const T* const h2e,
                    T* residuals) const
    {
        T ptn_pt[3] = {T(pattern_pt(0)), T(pattern_pt(1)), T(pattern_pt(2))},
                tcp_pt[3], base_r[3], cam_pt[3], cam_pt_pnp[3];
        T ptn_pt2[3] = {T(pattern_pt(0)), T(pattern_pt(1)), T(pattern_pt(2))};

        // Apply the current board2ee transformation
        ceres::AngleAxisRotatePoint(b2w, ptn_pt, base_r);

        base_r[0] += b2w[3];
        base_r[1] += b2w[4];
        base_r[2] += b2w[5];

        T robot_r[3] = {T(robot_r_vec(0)), T(robot_r_vec(1)), T(robot_r_vec(2))},
                robot_t[3] = {T(robot_t_vec(0)), T(robot_t_vec(1)), T(robot_t_vec(2))};

        // Apply the robot position transformation
        ceres::AngleAxisRotatePoint(robot_r, base_r, tcp_pt);

        tcp_pt[0] += robot_t[0];
        tcp_pt[1] += robot_t[1];
        tcp_pt[2] += robot_t[2];

        // Apply the current h2e transformation
        ceres::AngleAxisRotatePoint(h2e, tcp_pt, cam_pt);

        cam_pt[0] += h2e[3];
        cam_pt[1] += h2e[4];
        cam_pt[2] += h2e[5];

        //Chain2
        T pnp_r[3] = {T(pnp_r_vec(0)), T(pnp_r_vec(1)), T(pnp_r_vec(2))};
        T pnp_t[3] = {T(pnp_t_vec(0)), T(pnp_t_vec(1)), T(pnp_t_vec(2))};

        // Apply the current pnp transformation
        ceres::AngleAxisRotatePoint(pnp_r, ptn_pt2, cam_pt_pnp);

        cam_pt_pnp[0] += pnp_t[0];
        cam_pt_pnp[1] += pnp_t[1];
        cam_pt_pnp[2] += pnp_t[2];

        residuals[0] = (cam_pt[0] - cam_pt_pnp[0]);
        residuals[1] = (cam_pt[1] - cam_pt_pnp[1]);
        residuals[2] = (cam_pt[2] - cam_pt_pnp[2]);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const Eigen::Vector3d &pnp_r_vec,
                                        const Eigen::Vector3d &pnp_t_vec,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec,
                                        const Eigen::Vector3d &pattern_pt)
    {
        return new ceres::AutoDiffCostFunction<Classic_pnp, 3, 6, 6>(
                new Classic_pnp(pnp_r_vec, pnp_t_vec, robot_r_vec, robot_t_vec, pattern_pt));
    }

    Eigen::Vector3d pnp_r_vec;
    Eigen::Vector3d pnp_t_vec;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
    Eigen::Vector3d pattern_pt;
};


struct AX_ZB
{
    AX_ZB(const Eigen::Vector3d &pnp_r_vec,
                  const Eigen::Vector3d &pnp_t_vec,
                  const Eigen::Vector3d &robot_r_vec,
                  const Eigen::Vector3d &robot_t_vec):
            pnp_r_vec(pnp_r_vec),
            pnp_t_vec(pnp_t_vec),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec){}

    //template <typename T>
    bool operator()(//const double* const weights,
            const double* const b2w,
            const double* const h2e,
            double* residuals) const
    {
        double tcp_pt[3], tcp_pt_opt[3], cal_pt[3], base_r[3], base_pt[3], rot_mat_h2e[9], rot_mat_inv_h2e[9], rot_vec_h2e[3], tras_vec_h2e[3],
                opt_rot_mat[9], opt_rot_mat_inv[9], rot_vec_opt[3], tras_vec_opt[3],
                rob_rot_mat[9], rob_rot_mat_inv[9], rot_vec_rob[3], tras_vec_rob[3],
                cal_rot_mat[9], cal_rot_mat_inv[9], rot_vec_cal[3], tras_vec_cal[3],
                rot_mat_inv_1[9], rot_vec_1[3], tras_vec_1[3], rot_mat_inv_2[9],
                rot_vec_2[3], tras_vec_2[3], b2w_rot_mat[9], h2e_inv[3], b2w_inv[1], tras_vec_3[3];

        ceres::AngleAxisToRotationMatrix(h2e, rot_mat_h2e);
        cv::Mat X = (cv::Mat_<double>(4,4) <<
                                           rot_mat_h2e[0], rot_mat_h2e[3], rot_mat_h2e[6], h2e[3],
                rot_mat_h2e[1], rot_mat_h2e[4], rot_mat_h2e[7], h2e[4],
                rot_mat_h2e[2], rot_mat_h2e[5], rot_mat_h2e[8], h2e[5],
                0, 0, 0, 1);

        cv::Mat X_inv = X.inv();

        double robot_r[3] = {double(robot_r_vec(0)), double(robot_r_vec(1)), double(robot_r_vec(2))},
                robot_t[3] = {double(robot_t_vec(0)), double(robot_t_vec(1)), double(robot_t_vec(2))};

        ceres::AngleAxisToRotationMatrix(robot_r, rob_rot_mat);
        cv::Mat A = (cv::Mat_<double>(4,4) <<
                                           rob_rot_mat[0], rob_rot_mat[3], rob_rot_mat[6], robot_t[0],
                rob_rot_mat[1], rob_rot_mat[4], rob_rot_mat[7], robot_t[1],
                rob_rot_mat[2], rob_rot_mat[5], rob_rot_mat[8], robot_t[2],
                0, 0, 0, 1);


        cv::Mat chain1 = A*X; // T^{W}_{R} * T^{R}_{C}

        rot_mat_inv_1[0] = chain1.at<double>(0,0);
        rot_mat_inv_1[1] = chain1.at<double>(1,0);
        rot_mat_inv_1[2] = chain1.at<double>(2,0);
        rot_mat_inv_1[3] = chain1.at<double>(0,1);
        rot_mat_inv_1[4] = chain1.at<double>(1,1);
        rot_mat_inv_1[5] = chain1.at<double>(2,1);
        rot_mat_inv_1[6] = chain1.at<double>(0,2);
        rot_mat_inv_1[7] = chain1.at<double>(1,2);
        rot_mat_inv_1[8] = chain1.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_1, rot_vec_1);

        tras_vec_1[0] = chain1.at<double>(0,3);
        tras_vec_1[1] = chain1.at<double>(1,3);
        tras_vec_1[2] = chain1.at<double>(2,3);
        //###########################################################################################

        // Chain 2
        double pnp_r[3] = {double(pnp_r_vec(0)), double(pnp_r_vec(1)), double(pnp_r_vec(2))};
        double pnp_t[3] = {double(pnp_t_vec(0)), double(pnp_t_vec(1)), double(pnp_t_vec(2))};

        ceres::AngleAxisToRotationMatrix(pnp_r, cal_rot_mat);
        cv::Mat B_inv = (cv::Mat_<double>(4,4) <<
                                               cal_rot_mat[0], cal_rot_mat[3], cal_rot_mat[6], pnp_t[0],
                cal_rot_mat[1], cal_rot_mat[4], cal_rot_mat[7], pnp_t[1],
                cal_rot_mat[2], cal_rot_mat[5], cal_rot_mat[8], pnp_t[2],
                0, 0, 0, 1);

        cv::Mat B = B_inv.inv();

        ceres::AngleAxisToRotationMatrix(b2w, b2w_rot_mat);
        cv::Mat Z = (cv::Mat_<double>(4,4) <<
                                           b2w_rot_mat[0], b2w_rot_mat[3], b2w_rot_mat[6], b2w[3],
                b2w_rot_mat[1], b2w_rot_mat[4], b2w_rot_mat[7], b2w[4],
                b2w_rot_mat[2], b2w_rot_mat[5], b2w_rot_mat[8], b2w[5],
                0, 0, 0, 1);

        cv::Mat chain2 = Z*B;

        rot_mat_inv_2[0] = chain2.at<double>(0,0);
        rot_mat_inv_2[1] = chain2.at<double>(1,0);
        rot_mat_inv_2[2] = chain2.at<double>(2,0);
        rot_mat_inv_2[3] = chain2.at<double>(0,1);
        rot_mat_inv_2[4] = chain2.at<double>(1,1);
        rot_mat_inv_2[5] = chain2.at<double>(2,1);
        rot_mat_inv_2[6] = chain2.at<double>(0,2);
        rot_mat_inv_2[7] = chain2.at<double>(1,2);
        rot_mat_inv_2[8] = chain2.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_2, rot_vec_2);

        tras_vec_2[0] = chain2.at<double>(0,3);
        tras_vec_2[1] = chain2.at<double>(1,3);
        tras_vec_2[2] = chain2.at<double>(2,3);
        //###########################################################################################


        residuals[0] = (tras_vec_2[0] - tras_vec_1[0]);
        residuals[1] = (tras_vec_2[1] - tras_vec_1[1]);
        residuals[2] = (tras_vec_2[2] - tras_vec_1[2]);
        residuals[3] = (rot_vec_2[0] - rot_vec_1[0]);
        residuals[4] = (rot_vec_2[1] - rot_vec_1[1]);
        residuals[5] = (rot_vec_2[2] - rot_vec_1[2]);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const Eigen::Vector3d &pnp_r_vec,
                                        const Eigen::Vector3d &pnp_t_vec,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec)
    {
        return new ceres::NumericDiffCostFunction<AX_ZB, ceres::CENTRAL, 6, 6, 6>(
                new AX_ZB(pnp_r_vec, pnp_t_vec, robot_r_vec, robot_t_vec));
    }

    Eigen::Vector3d pnp_r_vec;
    Eigen::Vector3d pnp_t_vec;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
};



struct XBi_AiZ
{
    XBi_AiZ(const Eigen::Vector3d &pnp_r_vec,
          const Eigen::Vector3d &pnp_t_vec,
          const Eigen::Vector3d &robot_r_vec,
          const Eigen::Vector3d &robot_t_vec):
            pnp_r_vec(pnp_r_vec),
            pnp_t_vec(pnp_t_vec),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec){}

    //template <typename T>
    bool operator()(//const double* const weights,
            const double* const b2w,
            const double* const h2e,
            double* residuals) const
    {
        double tcp_pt[3], tcp_pt_opt[3], cal_pt[3], base_r[3], base_pt[3], rot_mat_h2e[9], rot_mat_inv_h2e[9], rot_vec_h2e[3], tras_vec_h2e[3],
                opt_rot_mat[9], opt_rot_mat_inv[9], rot_vec_opt[3], tras_vec_opt[3],
                rob_rot_mat[9], rob_rot_mat_inv[9], rot_vec_rob[3], tras_vec_rob[3],
                cal_rot_mat[9], cal_rot_mat_inv[9], rot_vec_cal[3], tras_vec_cal[3],
                rot_mat_inv_1[9], rot_vec_1[3], tras_vec_1[3], rot_mat_inv_2[9],
                rot_vec_2[3], tras_vec_2[3], b2w_rot_mat[9], h2e_inv[3], b2w_inv[1], tras_vec_3[3];

        ceres::AngleAxisToRotationMatrix(h2e, rot_mat_h2e);
        cv::Mat X = (cv::Mat_<double>(4,4) <<
                                           rot_mat_h2e[0], rot_mat_h2e[3], rot_mat_h2e[6], h2e[3],
                rot_mat_h2e[1], rot_mat_h2e[4], rot_mat_h2e[7], h2e[4],
                rot_mat_h2e[2], rot_mat_h2e[5], rot_mat_h2e[8], h2e[5],
                0, 0, 0, 1);

        cv::Mat X_inv = X.inv();

        double robot_r[3] = {double(robot_r_vec(0)), double(robot_r_vec(1)), double(robot_r_vec(2))},
                robot_t[3] = {double(robot_t_vec(0)), double(robot_t_vec(1)), double(robot_t_vec(2))};

        ceres::AngleAxisToRotationMatrix(robot_r, rob_rot_mat);
        cv::Mat A = (cv::Mat_<double>(4,4) <<
                                           rob_rot_mat[0], rob_rot_mat[3], rob_rot_mat[6], robot_t[0],
                rob_rot_mat[1], rob_rot_mat[4], rob_rot_mat[7], robot_t[1],
                rob_rot_mat[2], rob_rot_mat[5], rob_rot_mat[8], robot_t[2],
                0, 0, 0, 1);



        //###########################################################################################

        // Chain 2
        double pnp_r[3] = {double(pnp_r_vec(0)), double(pnp_r_vec(1)), double(pnp_r_vec(2))};
        double pnp_t[3] = {double(pnp_t_vec(0)), double(pnp_t_vec(1)), double(pnp_t_vec(2))};

        ceres::AngleAxisToRotationMatrix(pnp_r, cal_rot_mat);
        cv::Mat B_inv = (cv::Mat_<double>(4,4) <<
                                               cal_rot_mat[0], cal_rot_mat[3], cal_rot_mat[6], pnp_t[0],
                cal_rot_mat[1], cal_rot_mat[4], cal_rot_mat[7], pnp_t[1],
                cal_rot_mat[2], cal_rot_mat[5], cal_rot_mat[8], pnp_t[2],
                0, 0, 0, 1);

        cv::Mat B = B_inv.inv();

        ceres::AngleAxisToRotationMatrix(b2w, b2w_rot_mat);
        cv::Mat Z = (cv::Mat_<double>(4,4) <<
                                           b2w_rot_mat[0], b2w_rot_mat[3], b2w_rot_mat[6], b2w[3],
                b2w_rot_mat[1], b2w_rot_mat[4], b2w_rot_mat[7], b2w[4],
                b2w_rot_mat[2], b2w_rot_mat[5], b2w_rot_mat[8], b2w[5],
                0, 0, 0, 1);

        cv::Mat chain1 = X*B.inv(); // T^{W}_{R} * T^{R}_{C}
        cv::Mat chain2 = A.inv()*Z;

        rot_mat_inv_1[0] = chain1.at<double>(0,0);
        rot_mat_inv_1[1] = chain1.at<double>(1,0);
        rot_mat_inv_1[2] = chain1.at<double>(2,0);
        rot_mat_inv_1[3] = chain1.at<double>(0,1);
        rot_mat_inv_1[4] = chain1.at<double>(1,1);
        rot_mat_inv_1[5] = chain1.at<double>(2,1);
        rot_mat_inv_1[6] = chain1.at<double>(0,2);
        rot_mat_inv_1[7] = chain1.at<double>(1,2);
        rot_mat_inv_1[8] = chain1.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_1, rot_vec_1);

        tras_vec_1[0] = chain1.at<double>(0,3);
        tras_vec_1[1] = chain1.at<double>(1,3);
        tras_vec_1[2] = chain1.at<double>(2,3);

        rot_mat_inv_2[0] = chain2.at<double>(0,0);
        rot_mat_inv_2[1] = chain2.at<double>(1,0);
        rot_mat_inv_2[2] = chain2.at<double>(2,0);
        rot_mat_inv_2[3] = chain2.at<double>(0,1);
        rot_mat_inv_2[4] = chain2.at<double>(1,1);
        rot_mat_inv_2[5] = chain2.at<double>(2,1);
        rot_mat_inv_2[6] = chain2.at<double>(0,2);
        rot_mat_inv_2[7] = chain2.at<double>(1,2);
        rot_mat_inv_2[8] = chain2.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_2, rot_vec_2);

        tras_vec_2[0] = chain2.at<double>(0,3);
        tras_vec_2[1] = chain2.at<double>(1,3);
        tras_vec_2[2] = chain2.at<double>(2,3);
        //###########################################################################################


        residuals[0] = (tras_vec_2[0] - tras_vec_1[0]);
        residuals[1] = (tras_vec_2[1] - tras_vec_1[1]);
        residuals[2] = (tras_vec_2[2] - tras_vec_1[2]);
        residuals[3] = (rot_vec_2[0] - rot_vec_1[0]);
        residuals[4] = (rot_vec_2[1] - rot_vec_1[1]);
        residuals[5] = (rot_vec_2[2] - rot_vec_1[2]);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const Eigen::Vector3d &pnp_r_vec,
                                        const Eigen::Vector3d &pnp_t_vec,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec)
    {
        return new ceres::NumericDiffCostFunction<XBi_AiZ, ceres::CENTRAL, 6, 6, 6>(
                new XBi_AiZ(pnp_r_vec, pnp_t_vec, robot_r_vec, robot_t_vec));
    }

    Eigen::Vector3d pnp_r_vec;
    Eigen::Vector3d pnp_t_vec;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
};


struct ZiA_BXi
{
    ZiA_BXi(const Eigen::Vector3d &pnp_r_vec,
            const Eigen::Vector3d &pnp_t_vec,
            const Eigen::Vector3d &robot_r_vec,
            const Eigen::Vector3d &robot_t_vec):
            pnp_r_vec(pnp_r_vec),
            pnp_t_vec(pnp_t_vec),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec){}

    //template <typename T>
    bool operator()(//const double* const weights,
            const double* const b2w,
            const double* const h2e,
            double* residuals) const
    {
        double tcp_pt[3], tcp_pt_opt[3], cal_pt[3], base_r[3], base_pt[3], rot_mat_h2e[9], rot_mat_inv_h2e[9], rot_vec_h2e[3], tras_vec_h2e[3],
                opt_rot_mat[9], opt_rot_mat_inv[9], rot_vec_opt[3], tras_vec_opt[3],
                rob_rot_mat[9], rob_rot_mat_inv[9], rot_vec_rob[3], tras_vec_rob[3],
                cal_rot_mat[9], cal_rot_mat_inv[9], rot_vec_cal[3], tras_vec_cal[3],
                rot_mat_inv_1[9], rot_vec_1[3], tras_vec_1[3], rot_mat_inv_2[9],
                rot_vec_2[3], tras_vec_2[3], b2w_rot_mat[9], h2e_inv[3], b2w_inv[1], tras_vec_3[3];

        ceres::AngleAxisToRotationMatrix(h2e, rot_mat_h2e);
        cv::Mat X = (cv::Mat_<double>(4,4) <<
                                           rot_mat_h2e[0], rot_mat_h2e[3], rot_mat_h2e[6], h2e[3],
                rot_mat_h2e[1], rot_mat_h2e[4], rot_mat_h2e[7], h2e[4],
                rot_mat_h2e[2], rot_mat_h2e[5], rot_mat_h2e[8], h2e[5],
                0, 0, 0, 1);

        cv::Mat X_inv = X.inv();

        double robot_r[3] = {double(robot_r_vec(0)), double(robot_r_vec(1)), double(robot_r_vec(2))},
                robot_t[3] = {double(robot_t_vec(0)), double(robot_t_vec(1)), double(robot_t_vec(2))};

        ceres::AngleAxisToRotationMatrix(robot_r, rob_rot_mat);
        cv::Mat A = (cv::Mat_<double>(4,4) <<
                                           rob_rot_mat[0], rob_rot_mat[3], rob_rot_mat[6], robot_t[0],
                rob_rot_mat[1], rob_rot_mat[4], rob_rot_mat[7], robot_t[1],
                rob_rot_mat[2], rob_rot_mat[5], rob_rot_mat[8], robot_t[2],
                0, 0, 0, 1);



        //###########################################################################################

        // Chain 2
        double pnp_r[3] = {double(pnp_r_vec(0)), double(pnp_r_vec(1)), double(pnp_r_vec(2))};
        double pnp_t[3] = {double(pnp_t_vec(0)), double(pnp_t_vec(1)), double(pnp_t_vec(2))};

        ceres::AngleAxisToRotationMatrix(pnp_r, cal_rot_mat);
        cv::Mat B_inv = (cv::Mat_<double>(4,4) <<
                                               cal_rot_mat[0], cal_rot_mat[3], cal_rot_mat[6], pnp_t[0],
                cal_rot_mat[1], cal_rot_mat[4], cal_rot_mat[7], pnp_t[1],
                cal_rot_mat[2], cal_rot_mat[5], cal_rot_mat[8], pnp_t[2],
                0, 0, 0, 1);

        cv::Mat B = B_inv.inv();

        ceres::AngleAxisToRotationMatrix(b2w, b2w_rot_mat);
        cv::Mat Z = (cv::Mat_<double>(4,4) <<
                                           b2w_rot_mat[0], b2w_rot_mat[3], b2w_rot_mat[6], b2w[3],
                b2w_rot_mat[1], b2w_rot_mat[4], b2w_rot_mat[7], b2w[4],
                b2w_rot_mat[2], b2w_rot_mat[5], b2w_rot_mat[8], b2w[5],
                0, 0, 0, 1);

        cv::Mat chain1 = Z.inv()*A; // T^{W}_{R} * T^{R}_{C}
        cv::Mat chain2 = B*X.inv();

        rot_mat_inv_1[0] = chain1.at<double>(0,0);
        rot_mat_inv_1[1] = chain1.at<double>(1,0);
        rot_mat_inv_1[2] = chain1.at<double>(2,0);
        rot_mat_inv_1[3] = chain1.at<double>(0,1);
        rot_mat_inv_1[4] = chain1.at<double>(1,1);
        rot_mat_inv_1[5] = chain1.at<double>(2,1);
        rot_mat_inv_1[6] = chain1.at<double>(0,2);
        rot_mat_inv_1[7] = chain1.at<double>(1,2);
        rot_mat_inv_1[8] = chain1.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_1, rot_vec_1);

        tras_vec_1[0] = chain1.at<double>(0,3);
        tras_vec_1[1] = chain1.at<double>(1,3);
        tras_vec_1[2] = chain1.at<double>(2,3);

        rot_mat_inv_2[0] = chain2.at<double>(0,0);
        rot_mat_inv_2[1] = chain2.at<double>(1,0);
        rot_mat_inv_2[2] = chain2.at<double>(2,0);
        rot_mat_inv_2[3] = chain2.at<double>(0,1);
        rot_mat_inv_2[4] = chain2.at<double>(1,1);
        rot_mat_inv_2[5] = chain2.at<double>(2,1);
        rot_mat_inv_2[6] = chain2.at<double>(0,2);
        rot_mat_inv_2[7] = chain2.at<double>(1,2);
        rot_mat_inv_2[8] = chain2.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_2, rot_vec_2);

        tras_vec_2[0] = chain2.at<double>(0,3);
        tras_vec_2[1] = chain2.at<double>(1,3);
        tras_vec_2[2] = chain2.at<double>(2,3);
        //###########################################################################################


        residuals[0] = (tras_vec_2[0] - tras_vec_1[0]);
        residuals[1] = (tras_vec_2[1] - tras_vec_1[1]);
        residuals[2] = (tras_vec_2[2] - tras_vec_1[2]);
        residuals[3] = (rot_vec_2[0] - rot_vec_1[0]);
        residuals[4] = (rot_vec_2[1] - rot_vec_1[1]);
        residuals[5] = (rot_vec_2[2] - rot_vec_1[2]);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const Eigen::Vector3d &pnp_r_vec,
                                        const Eigen::Vector3d &pnp_t_vec,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec)
    {
        return new ceres::NumericDiffCostFunction<ZiA_BXi, ceres::CENTRAL, 6, 6, 6>(
                new ZiA_BXi(pnp_r_vec, pnp_t_vec, robot_r_vec, robot_t_vec));
    }

    Eigen::Vector3d pnp_r_vec;
    Eigen::Vector3d pnp_t_vec;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
};

struct BiZi_XiAi
{
    BiZi_XiAi(const Eigen::Vector3d &pnp_r_vec,
            const Eigen::Vector3d &pnp_t_vec,
            const Eigen::Vector3d &robot_r_vec,
            const Eigen::Vector3d &robot_t_vec):
            pnp_r_vec(pnp_r_vec),
            pnp_t_vec(pnp_t_vec),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec){}

    //template <typename T>
    bool operator()(//const double* const weights,
            const double* const b2w,
            const double* const h2e,
            double* residuals) const
    {
        double tcp_pt[3], tcp_pt_opt[3], cal_pt[3], base_r[3], base_pt[3], rot_mat_h2e[9], rot_mat_inv_h2e[9], rot_vec_h2e[3], tras_vec_h2e[3],
                opt_rot_mat[9], opt_rot_mat_inv[9], rot_vec_opt[3], tras_vec_opt[3],
                rob_rot_mat[9], rob_rot_mat_inv[9], rot_vec_rob[3], tras_vec_rob[3],
                cal_rot_mat[9], cal_rot_mat_inv[9], rot_vec_cal[3], tras_vec_cal[3],
                rot_mat_inv_1[9], rot_vec_1[3], tras_vec_1[3], rot_mat_inv_2[9],
                rot_vec_2[3], tras_vec_2[3], b2w_rot_mat[9], h2e_inv[3], b2w_inv[1], tras_vec_3[3];

        ceres::AngleAxisToRotationMatrix(h2e, rot_mat_h2e);
        cv::Mat X = (cv::Mat_<double>(4,4) <<
                                           rot_mat_h2e[0], rot_mat_h2e[3], rot_mat_h2e[6], h2e[3],
                rot_mat_h2e[1], rot_mat_h2e[4], rot_mat_h2e[7], h2e[4],
                rot_mat_h2e[2], rot_mat_h2e[5], rot_mat_h2e[8], h2e[5],
                0, 0, 0, 1);

        cv::Mat X_inv = X.inv();

        double robot_r[3] = {double(robot_r_vec(0)), double(robot_r_vec(1)), double(robot_r_vec(2))},
                robot_t[3] = {double(robot_t_vec(0)), double(robot_t_vec(1)), double(robot_t_vec(2))};

        ceres::AngleAxisToRotationMatrix(robot_r, rob_rot_mat);
        cv::Mat A = (cv::Mat_<double>(4,4) <<
                                           rob_rot_mat[0], rob_rot_mat[3], rob_rot_mat[6], robot_t[0],
                rob_rot_mat[1], rob_rot_mat[4], rob_rot_mat[7], robot_t[1],
                rob_rot_mat[2], rob_rot_mat[5], rob_rot_mat[8], robot_t[2],
                0, 0, 0, 1);



        //###########################################################################################

        // Chain 2
        double pnp_r[3] = {double(pnp_r_vec(0)), double(pnp_r_vec(1)), double(pnp_r_vec(2))};
        double pnp_t[3] = {double(pnp_t_vec(0)), double(pnp_t_vec(1)), double(pnp_t_vec(2))};

        ceres::AngleAxisToRotationMatrix(pnp_r, cal_rot_mat);
        cv::Mat B_inv = (cv::Mat_<double>(4,4) <<
                                               cal_rot_mat[0], cal_rot_mat[3], cal_rot_mat[6], pnp_t[0],
                cal_rot_mat[1], cal_rot_mat[4], cal_rot_mat[7], pnp_t[1],
                cal_rot_mat[2], cal_rot_mat[5], cal_rot_mat[8], pnp_t[2],
                0, 0, 0, 1);

        cv::Mat B = B_inv.inv();

        ceres::AngleAxisToRotationMatrix(b2w, b2w_rot_mat);
        cv::Mat Z = (cv::Mat_<double>(4,4) <<
                                           b2w_rot_mat[0], b2w_rot_mat[3], b2w_rot_mat[6], b2w[3],
                b2w_rot_mat[1], b2w_rot_mat[4], b2w_rot_mat[7], b2w[4],
                b2w_rot_mat[2], b2w_rot_mat[5], b2w_rot_mat[8], b2w[5],
                0, 0, 0, 1);

        cv::Mat chain1 = X.inv()*A.inv(); // T^{W}_{R} * T^{R}_{C}
        cv::Mat chain2 = B.inv()*Z.inv();

        rot_mat_inv_1[0] = chain1.at<double>(0,0);
        rot_mat_inv_1[1] = chain1.at<double>(1,0);
        rot_mat_inv_1[2] = chain1.at<double>(2,0);
        rot_mat_inv_1[3] = chain1.at<double>(0,1);
        rot_mat_inv_1[4] = chain1.at<double>(1,1);
        rot_mat_inv_1[5] = chain1.at<double>(2,1);
        rot_mat_inv_1[6] = chain1.at<double>(0,2);
        rot_mat_inv_1[7] = chain1.at<double>(1,2);
        rot_mat_inv_1[8] = chain1.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_1, rot_vec_1);

        tras_vec_1[0] = chain1.at<double>(0,3);
        tras_vec_1[1] = chain1.at<double>(1,3);
        tras_vec_1[2] = chain1.at<double>(2,3);

        rot_mat_inv_2[0] = chain2.at<double>(0,0);
        rot_mat_inv_2[1] = chain2.at<double>(1,0);
        rot_mat_inv_2[2] = chain2.at<double>(2,0);
        rot_mat_inv_2[3] = chain2.at<double>(0,1);
        rot_mat_inv_2[4] = chain2.at<double>(1,1);
        rot_mat_inv_2[5] = chain2.at<double>(2,1);
        rot_mat_inv_2[6] = chain2.at<double>(0,2);
        rot_mat_inv_2[7] = chain2.at<double>(1,2);
        rot_mat_inv_2[8] = chain2.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_2, rot_vec_2);

        tras_vec_2[0] = chain2.at<double>(0,3);
        tras_vec_2[1] = chain2.at<double>(1,3);
        tras_vec_2[2] = chain2.at<double>(2,3);
        //###########################################################################################


        residuals[0] = (tras_vec_2[0] - tras_vec_1[0]);
        residuals[1] = (tras_vec_2[1] - tras_vec_1[1]);
        residuals[2] = (tras_vec_2[2] - tras_vec_1[2]);
        residuals[3] = (rot_vec_2[0] - rot_vec_1[0]);
        residuals[4] = (rot_vec_2[1] - rot_vec_1[1]);
        residuals[5] = (rot_vec_2[2] - rot_vec_1[2]);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const Eigen::Vector3d &pnp_r_vec,
                                        const Eigen::Vector3d &pnp_t_vec,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec)
    {
        return new ceres::NumericDiffCostFunction<BiZi_XiAi, ceres::CENTRAL, 6, 6, 6>(
                new BiZi_XiAi(pnp_r_vec, pnp_t_vec, robot_r_vec, robot_t_vec));
    }

    Eigen::Vector3d pnp_r_vec;
    Eigen::Vector3d pnp_t_vec;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
};


struct Classic_pnp_2
{
    Classic_pnp_2(const Eigen::Vector3d &pnp_r_vec,
                const Eigen::Vector3d &pnp_t_vec,
                const Eigen::Vector3d &robot_r_vec,
                const Eigen::Vector3d &robot_t_vec,
                const double &tz,
                const double &camz):
            pnp_r_vec(pnp_r_vec),
            pnp_t_vec(pnp_t_vec),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec),
            tz(tz),
            camz(camz){}

    //template <typename T>
    bool operator()(//const double* const weights,
            const double* const b2w,
            const double* const h2e,
            double* residuals) const
    {
        //T ptn_pt[3] = {T(pattern_pt(0)), T(pattern_pt(1)), T(pattern_pt(2))},
        //  tcp_pt[3], base_r[3], cam_pt[3];
        //double cam_pt[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))},
        double tcp_pt[3], tcp_pt_opt[3], cal_pt[3], base_r[3], base_pt[3], rot_mat_h2e[9], rot_mat_inv_h2e[9], rot_vec_h2e[3], tras_vec_h2e[3],
                opt_rot_mat[9], opt_rot_mat_inv[9], rot_vec_opt[3], tras_vec_opt[3],
                rob_rot_mat[9], rob_rot_mat_inv[9], rot_vec_rob[3], tras_vec_rob[3],
                cal_rot_mat[9], cal_rot_mat_inv[9], rot_vec_cal[3], tras_vec_cal[3],
                rot_mat_inv_1[9], rot_vec_1[3], tras_vec_1[3], rot_mat_inv_2[9],
                rot_vec_2[3], tras_vec_2[3], b2w_rot_mat[9], h2e_inv[3], b2w_inv[1], tras_vec_3[3];
        //double cam_pt2[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))};
        double quaternion_opt[4];

        ceres::AngleAxisToRotationMatrix(h2e, rot_mat_h2e);
        cv::Mat X = (cv::Mat_<double>(4,4) <<
                rot_mat_h2e[0], rot_mat_h2e[3], rot_mat_h2e[6], h2e[3],
                rot_mat_h2e[1], rot_mat_h2e[4], rot_mat_h2e[7], h2e[4],
                rot_mat_h2e[2], rot_mat_h2e[5], rot_mat_h2e[8], h2e[5],
                0, 0, 0, 1);

        cv::Mat X_inv = X.inv();

        double robot_r[3] = {double(robot_r_vec(0)), double(robot_r_vec(1)), double(robot_r_vec(2))},
                robot_t[3] = {double(robot_t_vec(0)), double(robot_t_vec(1)), double(robot_t_vec(2))};

        ceres::AngleAxisToRotationMatrix(robot_r, rob_rot_mat);
        cv::Mat A = (cv::Mat_<double>(4,4) <<
                rob_rot_mat[0], rob_rot_mat[3], rob_rot_mat[6], robot_t[0],
                rob_rot_mat[1], rob_rot_mat[4], rob_rot_mat[7], robot_t[1],
                rob_rot_mat[2], rob_rot_mat[5], rob_rot_mat[8], robot_t[2],
                0, 0, 0, 1);


        cv::Mat chain1 = A*X; // T^{W}_{R} * T^{R}_{C}

        rot_mat_inv_1[0] = chain1.at<double>(0,0);
        rot_mat_inv_1[1] = chain1.at<double>(1,0);
        rot_mat_inv_1[2] = chain1.at<double>(2,0);
        rot_mat_inv_1[3] = chain1.at<double>(0,1);
        rot_mat_inv_1[4] = chain1.at<double>(1,1);
        rot_mat_inv_1[5] = chain1.at<double>(2,1);
        rot_mat_inv_1[6] = chain1.at<double>(0,2);
        rot_mat_inv_1[7] = chain1.at<double>(1,2);
        rot_mat_inv_1[8] = chain1.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_1, rot_vec_1);

        tras_vec_1[0] = chain1.at<double>(0,3);
        tras_vec_1[1] = chain1.at<double>(1,3);
        tras_vec_1[2] = chain1.at<double>(2,3);
        //###########################################################################################

        // Chain 2
        double pnp_r[3] = {double(pnp_r_vec(0)), double(pnp_r_vec(1)), double(pnp_r_vec(2))};
        double pnp_t[3] = {double(pnp_t_vec(0)), double(pnp_t_vec(1)), double(pnp_t_vec(2))};

        ceres::AngleAxisToRotationMatrix(pnp_r, cal_rot_mat);
        cv::Mat B_inv = (cv::Mat_<double>(4,4) <<
                cal_rot_mat[0], cal_rot_mat[3], cal_rot_mat[6], pnp_t[0],
                cal_rot_mat[1], cal_rot_mat[4], cal_rot_mat[7], pnp_t[1],
                cal_rot_mat[2], cal_rot_mat[5], cal_rot_mat[8], pnp_t[2],
                0, 0, 0, 1);

        cv::Mat B = B_inv.inv();

        ceres::AngleAxisToRotationMatrix(b2w, b2w_rot_mat);
        cv::Mat Z = (cv::Mat_<double>(4,4) <<
                b2w_rot_mat[0], b2w_rot_mat[3], b2w_rot_mat[6], b2w[3],
                b2w_rot_mat[1], b2w_rot_mat[4], b2w_rot_mat[7], b2w[4],
                b2w_rot_mat[2], b2w_rot_mat[5], b2w_rot_mat[8], b2w[5],
                0, 0, 0, 1);

        cv::Mat chain2 = Z*B;

        rot_mat_inv_2[0] = chain2.at<double>(0,0);
        rot_mat_inv_2[1] = chain2.at<double>(1,0);
        rot_mat_inv_2[2] = chain2.at<double>(2,0);
        rot_mat_inv_2[3] = chain2.at<double>(0,1);
        rot_mat_inv_2[4] = chain2.at<double>(1,1);
        rot_mat_inv_2[5] = chain2.at<double>(2,1);
        rot_mat_inv_2[6] = chain2.at<double>(0,2);
        rot_mat_inv_2[7] = chain2.at<double>(1,2);
        rot_mat_inv_2[8] = chain2.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_2, rot_vec_2);

        tras_vec_2[0] = chain2.at<double>(0,3);
        tras_vec_2[1] = chain2.at<double>(1,3);
        tras_vec_2[2] = chain2.at<double>(2,3);
        //###########################################################################################


        residuals[0] = (tras_vec_2[0] - tras_vec_1[0]);
        residuals[1] = (tras_vec_2[1] - tras_vec_1[1]);
        residuals[2] = (tras_vec_2[2] - tras_vec_1[2]);
        residuals[3] = (rot_vec_2[0] - rot_vec_1[0]);
        residuals[4] = (rot_vec_2[1] - rot_vec_1[1]);
        residuals[5] = (rot_vec_2[2] - rot_vec_1[2]);
        residuals[6] = b2w[5] - abs(tz);
        residuals[7] = tras_vec_1[2] - abs(camz);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const Eigen::Vector3d &pnp_r_vec,
                                        const Eigen::Vector3d &pnp_t_vec,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec,
                                        double &tz,
                                        double &camz)
    {
        return new ceres::NumericDiffCostFunction<Classic_pnp_2, ceres::CENTRAL, 8, 6, 6>(
                new Classic_pnp_2(pnp_r_vec, pnp_t_vec, robot_r_vec, robot_t_vec, tz, camz));
    }

    Eigen::Vector3d pnp_r_vec;
    Eigen::Vector3d pnp_t_vec;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
    double tz;
    double camz;
};


struct Classic_pnp_3
{
    Classic_pnp_3(const Eigen::Vector3d &pnp_r_vec,
                  const Eigen::Vector3d &pnp_t_vec,
                  const Eigen::Vector3d &robot_r_vec,
                  const Eigen::Vector3d &robot_t_vec,
                  const double &tz,
                  const double &camz
                  ) :
            pnp_r_vec(pnp_r_vec),
            pnp_t_vec(pnp_t_vec),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec),
            tz(tz),
            camz(camz){}

    //template <typename T>
    bool operator()(//const double* const weights,
            const double* const b2w,
            const double* const h2e,
            double* residuals) const
    {
        //T ptn_pt[3] = {T(pattern_pt(0)), T(pattern_pt(1)), T(pattern_pt(2))},
        //  tcp_pt[3], base_r[3], cam_pt[3];
        //double ptn_pt[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))},
        double tcp_pt[3], tcp_pt_opt[3], tcp_pt2[3], base_r[3], cam_pt[3], cam_pt_pnp[3], rot_mat[9], rot_mat_inv_1[9], rot_mat_inv_2[9],
            rot_vec_1[3], tras_vec_1[3], rot_vec_2[3], tras_vec_2[3], tras_vec_3[3],
                rot_vec_rob[3], rot_vec_h2e[3], rot_vec_pnp[3], rot_mat_rob[9], rot_mat_h2e[9], rot_mat_pnp[9],
                rot_mat_opt[9], h2e_inv[3];

        double robot_r[3] = {double(robot_r_vec(0)), double(robot_r_vec(1)), double(robot_r_vec(2))},
                robot_t[3] = {double(robot_t_vec(0)), double(robot_t_vec(1)), double(robot_t_vec(2))};

        ceres::AngleAxisToRotationMatrix(robot_r, rot_mat_rob);
        cv::Mat A = (cv::Mat_<double>(4,4) <<
                rot_mat_rob[0], rot_mat_rob[3], rot_mat_rob[6], robot_t[0],
                rot_mat_rob[1], rot_mat_rob[4], rot_mat_rob[7], robot_t[1],
                rot_mat_rob[2], rot_mat_rob[5], rot_mat_rob[8], robot_t[2],
                0, 0, 0, 1);

        ceres::AngleAxisToRotationMatrix(h2e, rot_mat_h2e);
        cv::Mat X = (cv::Mat_<double>(4,4) <<
                rot_mat_h2e[0], rot_mat_h2e[3], rot_mat_h2e[6], h2e[3],
                rot_mat_h2e[1], rot_mat_h2e[4], rot_mat_h2e[7], h2e[4],
                rot_mat_h2e[2], rot_mat_h2e[5], rot_mat_h2e[8], h2e[5],
                0, 0, 0, 1);

        cv::Mat X_inv = X.inv();

        cv::Mat chain_1 = X_inv*A.inv();                // T^{C}_{E} * T^{E}_{W}
        rot_mat_inv_1[0] = chain_1.at<double>(0,0);
        rot_mat_inv_1[1] = chain_1.at<double>(1,0);
        rot_mat_inv_1[2] = chain_1.at<double>(2,0);
        rot_mat_inv_1[3] = chain_1.at<double>(0,1);
        rot_mat_inv_1[4] = chain_1.at<double>(1,1);
        rot_mat_inv_1[5] = chain_1.at<double>(2,1);
        rot_mat_inv_1[6] = chain_1.at<double>(0,2);
        rot_mat_inv_1[7] = chain_1.at<double>(1,2);
        rot_mat_inv_1[8] = chain_1.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_1, rot_vec_1);

        tras_vec_1[0] = chain_1.at<double>(0,3);
        tras_vec_1[1] = chain_1.at<double>(1,3);
        tras_vec_1[2] = chain_1.at<double>(2,3);

        cv::Mat chain_3 = A*X;
        tras_vec_3[0] = chain_3.at<double>(0,3);
        tras_vec_3[1] = chain_3.at<double>(1,3);
        tras_vec_3[2] = chain_3.at<double>(2,3);

        //Chain2
        //###########################################################################################
        ceres::AngleAxisToRotationMatrix(b2w, rot_mat);
        cv::Mat Z = (cv::Mat_<double>(4,4) <<
                rot_mat[0], rot_mat[3], rot_mat[6], b2w[3],
                rot_mat[1], rot_mat[4], rot_mat[7], b2w[4],
                rot_mat[2], rot_mat[5], rot_mat[8], b2w[5],
                0, 0, 0, 1);

        //cv::Mat temp_b2w_inv = temp_b2w.inv();

        //###########################################################################################

        double pnp_r[3] = {double(pnp_r_vec(0)), double(pnp_r_vec(1)), double(pnp_r_vec(2))};
        double pnp_t[3] = {double(pnp_t_vec(0)), double(pnp_t_vec(1)), double(pnp_t_vec(2))};

        // Apply the current pnp transformation
        ceres::AngleAxisToRotationMatrix(pnp_r, rot_mat_pnp);
        cv::Mat B_inv = (cv::Mat_<double>(4,4) <<
                rot_mat_pnp[0], rot_mat_pnp[3], rot_mat_pnp[6], pnp_t[0],
                rot_mat_pnp[1], rot_mat_pnp[4], rot_mat_pnp[7], pnp_t[1],
                rot_mat_pnp[2], rot_mat_pnp[5], rot_mat_pnp[8], pnp_t[2],
                0, 0, 0, 1);

        cv::Mat chain_2 = B_inv*Z.inv(); // T^{C}_{B} * T^{B}_{W}
        rot_mat_inv_2[0] = chain_2.at<double>(0,0);
        rot_mat_inv_2[1] = chain_2.at<double>(1,0);
        rot_mat_inv_2[2] = chain_2.at<double>(2,0);
        rot_mat_inv_2[3] = chain_2.at<double>(0,1);
        rot_mat_inv_2[4] = chain_2.at<double>(1,1);
        rot_mat_inv_2[5] = chain_2.at<double>(2,1);
        rot_mat_inv_2[6] = chain_2.at<double>(0,2);
        rot_mat_inv_2[7] = chain_2.at<double>(1,2);
        rot_mat_inv_2[8] = chain_2.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_2, rot_vec_2);

        tras_vec_2[0] = chain_2.at<double>(0,3);
        tras_vec_2[1] = chain_2.at<double>(1,3);
        tras_vec_2[2] = chain_2.at<double>(2,3);

        residuals[0] = (rot_vec_2[0] - rot_vec_1[0]);
        residuals[1] = (rot_vec_2[1] - rot_vec_1[1]);
        residuals[2] = (rot_vec_2[2] - rot_vec_1[2]);
        residuals[3] = (tras_vec_2[0] - tras_vec_1[0]);
        residuals[4] = (tras_vec_2[1] - tras_vec_1[1]);
        residuals[5] = (tras_vec_2[2] - tras_vec_1[2]);
        residuals[6] = b2w[5] - abs(tz);
        residuals[7] = tras_vec_3[2] - abs(camz);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const Eigen::Vector3d &pnp_r_vec,
                                        const Eigen::Vector3d &pnp_t_vec,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec,
                                        const double &tz,
                                        const double &camz)
    {
        return new ceres::NumericDiffCostFunction<Classic_pnp_3, ceres::CENTRAL, 8, 6, 6>(
                new Classic_pnp_3(pnp_r_vec, pnp_t_vec, robot_r_vec, robot_t_vec, tz, camz));
    }

    Eigen::Vector3d pnp_r_vec;
    Eigen::Vector3d pnp_t_vec;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
    double tz;
    double camz;
};


struct Classic_pnp_4
{
    Classic_pnp_4(const Eigen::Vector3d &pnp_r_vec,
                  const Eigen::Vector3d &pnp_t_vec,
                  const Eigen::Vector3d &robot_r_vec,
                  const Eigen::Vector3d &robot_t_vec,
                  const double &tz,
                  const double &camz):
            pnp_r_vec(pnp_r_vec),
            pnp_t_vec(pnp_t_vec),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec),
            tz(tz),
            camz(camz){}

    //template <typename T>
    bool operator()(//const double* const weights,
            const double* const b2w,
            const double* const h2e,
            double* residuals) const
    {
        //T ptn_pt[3] = {T(pattern_pt(0)), T(pattern_pt(1)), T(pattern_pt(2))},
        //  tcp_pt[3], base_r[3], cam_pt[3];
        //double cam_pt[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))},
        double tcp_pt[3], tcp_pt_opt[3], cal_pt[3], base_r[3], base_pt[3], rot_mat_h2e[9], rot_mat_inv_h2e[9], rot_vec_h2e[3], tras_vec_h2e[3],
                opt_rot_mat[9], opt_rot_mat_inv[9], rot_vec_opt[3], tras_vec_opt[3],
                rob_rot_mat[9], rob_rot_mat_inv[9], rot_vec_rob[3], tras_vec_rob[3],
                cal_rot_mat[9], cal_rot_mat_inv[9], rot_vec_cal[3], tras_vec_cal[3],
                rot_mat_inv_1[9], rot_vec_1[3], tras_vec_1[3], rot_mat_inv_2[9],
                rot_vec_2[3], tras_vec_2[3], tras_vec_3[3], b2w_rot_mat[9], h2e_inv[3], b2w_inv[1];
        //double cam_pt2[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))};
        double quaternion_opt[4];

        ceres::AngleAxisToRotationMatrix(h2e, rot_mat_h2e);
        cv::Mat X = (cv::Mat_<double>(4,4) <<
                                                  rot_mat_h2e[0], rot_mat_h2e[3], rot_mat_h2e[6], h2e[3],
                rot_mat_h2e[1], rot_mat_h2e[4], rot_mat_h2e[7], h2e[4],
                rot_mat_h2e[2], rot_mat_h2e[5], rot_mat_h2e[8], h2e[5],
                0, 0, 0, 1);

        cv::Mat X_inv = X.inv();

        double robot_r[3] = {double(robot_r_vec(0)), double(robot_r_vec(1)), double(robot_r_vec(2))},
                robot_t[3] = {double(robot_t_vec(0)), double(robot_t_vec(1)), double(robot_t_vec(2))};

        ceres::AngleAxisToRotationMatrix(robot_r, rob_rot_mat);
        cv::Mat A = (cv::Mat_<double>(4,4) <<
                                                  rob_rot_mat[0], rob_rot_mat[3], rob_rot_mat[6], robot_t[0],
                rob_rot_mat[1], rob_rot_mat[4], rob_rot_mat[7], robot_t[1],
                rob_rot_mat[2], rob_rot_mat[5], rob_rot_mat[8], robot_t[2],
                0, 0, 0, 1);

        cv::Mat A_inv = A.inv();


        //###########################################################################################

        // Chain 2
        double pnp_r[3] = {double(pnp_r_vec(0)), double(pnp_r_vec(1)), double(pnp_r_vec(2))};
        double pnp_t[3] = {double(pnp_t_vec(0)), double(pnp_t_vec(1)), double(pnp_t_vec(2))};

        ceres::AngleAxisToRotationMatrix(pnp_r, cal_rot_mat);
        cv::Mat B_inv = (cv::Mat_<double>(4,4) <<
                                                  cal_rot_mat[0], cal_rot_mat[3], cal_rot_mat[6], pnp_t[0],
                cal_rot_mat[1], cal_rot_mat[4], cal_rot_mat[7], pnp_t[1],
                cal_rot_mat[2], cal_rot_mat[5], cal_rot_mat[8], pnp_t[2],
                0, 0, 0, 1);

        cv::Mat B = B_inv.inv();

        ceres::AngleAxisToRotationMatrix(b2w, b2w_rot_mat);
        cv::Mat Z = (cv::Mat_<double>(4,4) <<
                                                  b2w_rot_mat[0], b2w_rot_mat[3], b2w_rot_mat[6], b2w[3],
                b2w_rot_mat[1], b2w_rot_mat[4], b2w_rot_mat[7], b2w[4],
                b2w_rot_mat[2], b2w_rot_mat[5], b2w_rot_mat[8], b2w[5],
                0, 0, 0, 1);



        //cv::Mat chain1 = temp_rob_inv*temp_h2e_inv; // T^{W}_{E} * T^{E}_{C}
        //cv::Mat chain2 = temp_b2w*temp_cal_inv; // T^{W}_{B} * T^{B}_{C}

        cv::Mat chain1 = Z.inv()*A; // T^{B}_{W} * T^{W}_{R}
        cv::Mat chain2 = B*X.inv(); // T^{B}_{C} * T^{C}_{R}
        cv::Mat chain3 = A*X;

        rot_mat_inv_1[0] = chain1.at<double>(0,0);
        rot_mat_inv_1[1] = chain1.at<double>(1,0);
        rot_mat_inv_1[2] = chain1.at<double>(2,0);
        rot_mat_inv_1[3] = chain1.at<double>(0,1);
        rot_mat_inv_1[4] = chain1.at<double>(1,1);
        rot_mat_inv_1[5] = chain1.at<double>(2,1);
        rot_mat_inv_1[6] = chain1.at<double>(0,2);
        rot_mat_inv_1[7] = chain1.at<double>(1,2);
        rot_mat_inv_1[8] = chain1.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_1, rot_vec_1);

        tras_vec_1[0] = chain1.at<double>(0,3);
        tras_vec_1[1] = chain1.at<double>(1,3);
        tras_vec_1[2] = chain1.at<double>(2,3);

        tras_vec_3[0] = chain3.at<double>(0,3);
        tras_vec_3[1] = chain3.at<double>(1,3);
        tras_vec_3[2] = chain3.at<double>(2,3);

        rot_mat_inv_2[0] = chain2.at<double>(0,0);
        rot_mat_inv_2[1] = chain2.at<double>(1,0);
        rot_mat_inv_2[2] = chain2.at<double>(2,0);
        rot_mat_inv_2[3] = chain2.at<double>(0,1);
        rot_mat_inv_2[4] = chain2.at<double>(1,1);
        rot_mat_inv_2[5] = chain2.at<double>(2,1);
        rot_mat_inv_2[6] = chain2.at<double>(0,2);
        rot_mat_inv_2[7] = chain2.at<double>(1,2);
        rot_mat_inv_2[8] = chain2.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_2, rot_vec_2);

        tras_vec_2[0] = chain2.at<double>(0,3);
        tras_vec_2[1] = chain2.at<double>(1,3);
        tras_vec_2[2] = chain2.at<double>(2,3);
        //###########################################################################################


        residuals[0] = (tras_vec_2[0] - tras_vec_1[0]);
        residuals[1] = (tras_vec_2[1] - tras_vec_1[1]);
        residuals[2] = (tras_vec_2[2] - tras_vec_1[2]); //+ 0.02 * h2e_inv[2]
        residuals[3] = (rot_vec_2[0] - rot_vec_1[0]);
        residuals[4] = (rot_vec_2[1] - rot_vec_1[1]);
        residuals[5] = (rot_vec_2[2] - rot_vec_1[2]);
        residuals[6] = b2w[5] - abs(tz);
        residuals[7] = tras_vec_3[2] - abs(camz);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const Eigen::Vector3d &pnp_r_vec,
                                        const Eigen::Vector3d &pnp_t_vec,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec,
                                        const double &tz,
                                        const double &camz)
    {
        return new ceres::NumericDiffCostFunction<Classic_pnp_4, ceres::CENTRAL, 8, 6, 6>(
                new Classic_pnp_4(pnp_r_vec, pnp_t_vec, robot_r_vec, robot_t_vec, tz, camz));
    }

    Eigen::Vector3d pnp_r_vec;
    Eigen::Vector3d pnp_t_vec;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
    double tz;
    double camz;
};


struct Classic_pnp_5
{
    Classic_pnp_5(const Eigen::Vector3d &pnp_r_vec,
                  const Eigen::Vector3d &pnp_t_vec,
                  const Eigen::Vector3d &robot_r_vec,
                  const Eigen::Vector3d &robot_t_vec,
                  const double &tz,
                  const double &camz):
            pnp_r_vec(pnp_r_vec),
            pnp_t_vec(pnp_t_vec),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec),
            tz(tz),
            camz(camz){}

    //template <typename T>
    bool operator()(//const double* const weights,
            const double* const b2w,
            const double* const h2e,
            double* residuals) const
    {
        //T ptn_pt[3] = {T(pattern_pt(0)), T(pattern_pt(1)), T(pattern_pt(2))},
        //  tcp_pt[3], base_r[3], cam_pt[3];
        //double cam_pt[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))},
        double tcp_pt[3], tcp_pt_opt[3], cal_pt[3], base_r[3], base_pt[3], rot_mat_h2e[9], rot_mat_inv_h2e[9], rot_vec_h2e[3], tras_vec_h2e[3],
                opt_rot_mat[9], opt_rot_mat_inv[9], rot_vec_opt[3], tras_vec_opt[3],
                rob_rot_mat[9], rob_rot_mat_inv[9], rot_vec_rob[3], tras_vec_rob[3],
                cal_rot_mat[9], cal_rot_mat_inv[9], rot_vec_cal[3], tras_vec_cal[3],
                rot_mat_inv_1[9], rot_vec_1[3], tras_vec_1[3], rot_mat_inv_2[9],
                rot_vec_2[3], tras_vec_2[3], tras_vec_3[3], b2w_rot_mat[9], h2e_inv[3], b2w_inv[1];
        //double cam_pt2[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))};
        double quaternion_opt[4];

        ceres::AngleAxisToRotationMatrix(h2e, rot_mat_h2e);
        cv::Mat X = (cv::Mat_<double>(4,4) <<
                                                  rot_mat_h2e[0], rot_mat_h2e[3], rot_mat_h2e[6], h2e[3],
                rot_mat_h2e[1], rot_mat_h2e[4], rot_mat_h2e[7], h2e[4],
                rot_mat_h2e[2], rot_mat_h2e[5], rot_mat_h2e[8], h2e[5],
                0, 0, 0, 1);

        cv::Mat X_inv = X.inv();

        double robot_r[3] = {double(robot_r_vec(0)), double(robot_r_vec(1)), double(robot_r_vec(2))},
                robot_t[3] = {double(robot_t_vec(0)), double(robot_t_vec(1)), double(robot_t_vec(2))};

        ceres::AngleAxisToRotationMatrix(robot_r, rob_rot_mat);
        cv::Mat A = (cv::Mat_<double>(4,4) <<
                                                  rob_rot_mat[0], rob_rot_mat[3], rob_rot_mat[6], robot_t[0],
                rob_rot_mat[1], rob_rot_mat[4], rob_rot_mat[7], robot_t[1],
                rob_rot_mat[2], rob_rot_mat[5], rob_rot_mat[8], robot_t[2],
                0, 0, 0, 1);

        cv::Mat A_inv = A.inv();




        //###########################################################################################

        // Chain 2
        double pnp_r[3] = {double(pnp_r_vec(0)), double(pnp_r_vec(1)), double(pnp_r_vec(2))};
        double pnp_t[3] = {double(pnp_t_vec(0)), double(pnp_t_vec(1)), double(pnp_t_vec(2))};

        ceres::AngleAxisToRotationMatrix(pnp_r, cal_rot_mat);
        cv::Mat B_inv = (cv::Mat_<double>(4,4) <<
                                                  cal_rot_mat[0], cal_rot_mat[3], cal_rot_mat[6], pnp_t[0],
                cal_rot_mat[1], cal_rot_mat[4], cal_rot_mat[7], pnp_t[1],
                cal_rot_mat[2], cal_rot_mat[5], cal_rot_mat[8], pnp_t[2],
                0, 0, 0, 1);

        cv::Mat B = B_inv.inv();

        ceres::AngleAxisToRotationMatrix(b2w, b2w_rot_mat);
        cv::Mat Z = (cv::Mat_<double>(4,4) <<
                b2w_rot_mat[0], b2w_rot_mat[3], b2w_rot_mat[6], b2w[3],
                b2w_rot_mat[1], b2w_rot_mat[4], b2w_rot_mat[7], b2w[4],
                b2w_rot_mat[2], b2w_rot_mat[5], b2w_rot_mat[8], b2w[5],
                0, 0, 0, 1);



        //cv::Mat chain1 = temp_rob_inv*temp_h2e_inv; // T^{W}_{E} * T^{E}_{C}
        //cv::Mat chain2 = temp_b2w*temp_cal_inv; // T^{W}_{B} * T^{B}_{C}

        cv::Mat chain1 = X*B_inv; // T^{R}_{C} * T^{C}_{B}
        cv::Mat chain2 = A.inv()*Z; // T^{R}_{W} * T^{C}_{B}
        cv::Mat chain3 = A*X;

        rot_mat_inv_1[0] = chain1.at<double>(0,0);
        rot_mat_inv_1[1] = chain1.at<double>(1,0);
        rot_mat_inv_1[2] = chain1.at<double>(2,0);
        rot_mat_inv_1[3] = chain1.at<double>(0,1);
        rot_mat_inv_1[4] = chain1.at<double>(1,1);
        rot_mat_inv_1[5] = chain1.at<double>(2,1);
        rot_mat_inv_1[6] = chain1.at<double>(0,2);
        rot_mat_inv_1[7] = chain1.at<double>(1,2);
        rot_mat_inv_1[8] = chain1.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_1, rot_vec_1);

        tras_vec_1[0] = chain1.at<double>(0,3);
        tras_vec_1[1] = chain1.at<double>(1,3);
        tras_vec_1[2] = chain1.at<double>(2,3);

        tras_vec_3[0] = chain3.at<double>(0,3);
        tras_vec_3[1] = chain3.at<double>(1,3);
        tras_vec_3[2] = chain3.at<double>(2,3);

        rot_mat_inv_2[0] = chain2.at<double>(0,0);
        rot_mat_inv_2[1] = chain2.at<double>(1,0);
        rot_mat_inv_2[2] = chain2.at<double>(2,0);
        rot_mat_inv_2[3] = chain2.at<double>(0,1);
        rot_mat_inv_2[4] = chain2.at<double>(1,1);
        rot_mat_inv_2[5] = chain2.at<double>(2,1);
        rot_mat_inv_2[6] = chain2.at<double>(0,2);
        rot_mat_inv_2[7] = chain2.at<double>(1,2);
        rot_mat_inv_2[8] = chain2.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_2, rot_vec_2);

        tras_vec_2[0] = chain2.at<double>(0,3);
        tras_vec_2[1] = chain2.at<double>(1,3);
        tras_vec_2[2] = chain2.at<double>(2,3);
        //###########################################################################################


        residuals[0] = (tras_vec_2[0] - tras_vec_1[0]);
        residuals[1] = (tras_vec_2[1] - tras_vec_1[1]);
        residuals[2] = (tras_vec_2[2] - tras_vec_1[2]); //+ 0.02 * h2e_inv[2]
        residuals[3] = (rot_vec_2[0] - rot_vec_1[0]);
        residuals[4] = (rot_vec_2[1] - rot_vec_1[1]);
        residuals[5] = (rot_vec_2[2] - rot_vec_1[2]);
        residuals[6] = b2w[5] - abs(tz);
        residuals[7] = tras_vec_3[2] - abs(camz);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const Eigen::Vector3d &pnp_r_vec,
                                        const Eigen::Vector3d &pnp_t_vec,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec,
                                        const double &tz,
                                        const double &camz)
    {
        return new ceres::NumericDiffCostFunction<Classic_pnp_5, ceres::CENTRAL, 8, 6, 6>(
                new Classic_pnp_5(pnp_r_vec, pnp_t_vec, robot_r_vec, robot_t_vec, tz, camz));
    }

    Eigen::Vector3d pnp_r_vec;
    Eigen::Vector3d pnp_t_vec;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
    double tz;
    double camz;
};


struct Classic_pnp_opt
{
    Classic_pnp_opt(const Eigen::Vector3d &pnp_r_vec,
                const Eigen::Vector3d &pnp_t_vec,
                const Eigen::Vector3d &robot_r_vec,
                const Eigen::Vector3d &robot_t_vec,
                const Eigen::Vector3d &pattern_pt) :
            pnp_r_vec(pnp_r_vec),
            pnp_t_vec(pnp_t_vec),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec),
            pattern_pt(pattern_pt) {}

    //template <typename T>
    bool operator()(const double* const b2w,
                    const double* const h2e,
                    const double* const robot_opt,
                    double* residuals) const
    {
        //T ptn_pt[3] = {T(pattern_pt(0)), T(pattern_pt(1)), T(pattern_pt(2))},
        //  tcp_pt[3], base_r[3], cam_pt[3];
        double ptn_pt[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))},
                tcp_pt[3], tcp_pt_opt[3], base_r[3], cam_pt[3], cam_pt_pnp[3], rot_mat[9], rot_mat_opt[9], rot_vec[3], tras_vec_opt[3];
        double ptn_pt2[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))};

        // Apply the current board2ee transformation
        ceres::AngleAxisRotatePoint(b2w, ptn_pt, base_r);

        base_r[0] += b2w[3];
        base_r[1] += b2w[4];
        base_r[2] += b2w[5];

        double robot_r[3] = {double(robot_r_vec(0)), double(robot_r_vec(1)), double(robot_r_vec(2))},
                robot_t[3] = {double(robot_t_vec(0)), double(robot_t_vec(1)), double(robot_t_vec(2))};


        // Apply the robot position transformation
        ceres::AngleAxisRotatePoint(robot_r, base_r, tcp_pt);

        tcp_pt[0] += robot_t[0];
        tcp_pt[1] += robot_t[1];
        tcp_pt[2] += robot_t[2];


        // Apply the robot position transformation
        ceres::AngleAxisRotatePoint(robot_opt, tcp_pt, tcp_pt_opt);

        tcp_pt_opt[0] += robot_opt[3];
        tcp_pt_opt[1] += robot_opt[4];
        tcp_pt_opt[2] += robot_opt[5];

        // Apply the current h2e transformation
        ceres::AngleAxisRotatePoint(h2e, tcp_pt_opt, cam_pt);

        cam_pt[0] += h2e[3];
        cam_pt[1] += h2e[4];
        cam_pt[2] += h2e[5];

        //Chain2
        double pnp_r[3] = {double(pnp_r_vec(0)), double(pnp_r_vec(1)), double(pnp_r_vec(2))};
        double pnp_t[3] = {double(pnp_t_vec(0)), double(pnp_t_vec(1)), double(pnp_t_vec(2))};

        // Apply the current pnp transformation
        ceres::AngleAxisRotatePoint(pnp_r, ptn_pt2, cam_pt_pnp);

        cam_pt_pnp[0] += pnp_t[0];
        cam_pt_pnp[1] += pnp_t[1];
        cam_pt_pnp[2] += pnp_t[2];

        residuals[0] = (cam_pt[0] - cam_pt_pnp[0]);
        residuals[1] = (cam_pt[1] - cam_pt_pnp[1]);
        residuals[2] = (cam_pt[2] - cam_pt_pnp[2]);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const Eigen::Vector3d &pnp_r_vec,
                                        const Eigen::Vector3d &pnp_t_vec,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec,
                                        const Eigen::Vector3d &pattern_pt)
    {
        return new ceres::NumericDiffCostFunction<Classic_pnp_opt, ceres::CENTRAL, 3, 6, 6, 6>(
                new Classic_pnp_opt(pnp_r_vec, pnp_t_vec, robot_r_vec, robot_t_vec, pattern_pt));
    }

    Eigen::Vector3d pnp_r_vec;
    Eigen::Vector3d pnp_t_vec;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
    Eigen::Vector3d pattern_pt;
};


struct Classic_pnp_opt_inv
{
    Classic_pnp_opt_inv(const Eigen::Vector3d &pnp_r_vec,
                    const Eigen::Vector3d &pnp_t_vec,
                    const Eigen::Vector3d &robot_r_vec,
                    const Eigen::Vector3d &robot_t_vec,
                    const Eigen::Vector3d &pattern_pt) :
            pnp_r_vec(pnp_r_vec),
            pnp_t_vec(pnp_t_vec),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec),
            pattern_pt(pattern_pt) {}

    //template <typename T>
    bool operator()(const double* const b2w,
                    const double* const h2e,
                    const double* const robot_opt,
                    double* residuals) const
    {
        //T ptn_pt[3] = {T(pattern_pt(0)), T(pattern_pt(1)), T(pattern_pt(2))},
        //  tcp_pt[3], base_r[3], cam_pt[3];
        double cam_pt[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))},
                tcp_pt[3], tcp_pt_opt[3], base_r[3], w_pt[3], cam_pt_pnp[3], h2e_rot_mat[9], rob_rot_mat[9], opt_rot_mat[9], h2e_rot_mat_inv[9], rot_mat_opt[9],
                rot_vec_h2e[3], tras_vec_h2e[3], rob_rot_mat_inv[9], rot_vec_rob[3], tras_vec_rob[3], opt_rot_mat_inv[9], rot_vec_opt[3], tras_vec_opt[3],
                cal_rot_mat[9], cal_rot_mat_inv[9], rot_vec_cal[3], tras_vec_cal[3], cal_pt[3],
                cal_rot_mat2[9], cal_rot_mat_inv2[9], rot_vec_cal2[3], tras_vec_cal2[3], cal_pt2[3];;
        double cam_pt2[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))};

        //###########################################################################################
        ceres::AngleAxisToRotationMatrix(h2e, h2e_rot_mat);

        cv::Mat temp_h2e = (cv::Mat_<double>(4,4) <<
                h2e_rot_mat[0], h2e_rot_mat[3], h2e_rot_mat[6], h2e[3],
                h2e_rot_mat[1], h2e_rot_mat[4], h2e_rot_mat[7], h2e[4],
                h2e_rot_mat[2], h2e_rot_mat[5], h2e_rot_mat[8], h2e[5],
                0, 0, 0, 1);

        cv::Mat temp_h2e_inv = temp_h2e.inv();

        h2e_rot_mat_inv[0] = temp_h2e_inv.at<double>(0,0);
        h2e_rot_mat_inv[1] = temp_h2e_inv.at<double>(1,0);
        h2e_rot_mat_inv[2] = temp_h2e_inv.at<double>(2,0);
        h2e_rot_mat_inv[3] = temp_h2e_inv.at<double>(0,1);
        h2e_rot_mat_inv[4] = temp_h2e_inv.at<double>(1,1);
        h2e_rot_mat_inv[5] = temp_h2e_inv.at<double>(2,1);
        h2e_rot_mat_inv[6] = temp_h2e_inv.at<double>(0,2);
        h2e_rot_mat_inv[7] = temp_h2e_inv.at<double>(1,2);
        h2e_rot_mat_inv[8] = temp_h2e_inv.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(h2e_rot_mat_inv, rot_vec_h2e);

        tras_vec_h2e[0] = temp_h2e_inv.at<double>(0,3);
        tras_vec_h2e[1] = temp_h2e_inv.at<double>(1,3);
        tras_vec_h2e[2] = temp_h2e_inv.at<double>(2,3);

        //###########################################################################################

        // Apply the current board2ee transformation
        ceres::AngleAxisRotatePoint(rot_vec_h2e, cam_pt, tcp_pt_opt);

        tcp_pt_opt[0] += tras_vec_h2e[0];
        tcp_pt_opt[1] += tras_vec_h2e[1];
        tcp_pt_opt[2] += tras_vec_h2e[2];



        //###########################################################################################
        ceres::AngleAxisToRotationMatrix(robot_opt, opt_rot_mat);

        cv::Mat temp_opt = (cv::Mat_<double>(4,4) <<
                  opt_rot_mat[0], opt_rot_mat[3], opt_rot_mat[6], robot_opt[3],
                opt_rot_mat[1], opt_rot_mat[4], opt_rot_mat[7], robot_opt[4],
                opt_rot_mat[2], opt_rot_mat[5], opt_rot_mat[8], robot_opt[5],
                0, 0, 0, 1);

        cv::Mat temp_opt_inv = temp_opt.inv();

        opt_rot_mat_inv[0] = temp_opt_inv.at<double>(0,0);
        opt_rot_mat_inv[1] = temp_opt_inv.at<double>(1,0);
        opt_rot_mat_inv[2] = temp_opt_inv.at<double>(2,0);
        opt_rot_mat_inv[3] = temp_opt_inv.at<double>(0,1);
        opt_rot_mat_inv[4] = temp_opt_inv.at<double>(1,1);
        opt_rot_mat_inv[5] = temp_opt_inv.at<double>(2,1);
        opt_rot_mat_inv[6] = temp_opt_inv.at<double>(0,2);
        opt_rot_mat_inv[7] = temp_opt_inv.at<double>(1,2);
        opt_rot_mat_inv[8] = temp_opt_inv.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(opt_rot_mat_inv, rot_vec_opt);

        tras_vec_opt[0] = temp_opt_inv.at<double>(0,3);
        tras_vec_opt[1] = temp_opt_inv.at<double>(1,3);
        tras_vec_opt[2] = temp_opt_inv.at<double>(2,3);

        //###########################################################################################

        // Apply the robot position transformation
        ceres::AngleAxisRotatePoint(rot_vec_opt, tcp_pt_opt, tcp_pt);

        tcp_pt[0] += tras_vec_opt[0];
        tcp_pt[1] += tras_vec_opt[1];
        tcp_pt[2] += tras_vec_opt[2];

        double robot_r[3] = {double(robot_r_vec(0)), double(robot_r_vec(1)), double(robot_r_vec(2))},
               robot_t[3] = {double(robot_t_vec(0)), double(robot_t_vec(1)), double(robot_t_vec(2))};

        //###########################################################################################
        ceres::AngleAxisToRotationMatrix(robot_r, rob_rot_mat);

        cv::Mat temp_rob = (cv::Mat_<double>(4,4) <<
                rob_rot_mat[0], rob_rot_mat[3], rob_rot_mat[6], robot_t[0],
                rob_rot_mat[1], rob_rot_mat[4], rob_rot_mat[7], robot_t[1],
                rob_rot_mat[2], rob_rot_mat[5], rob_rot_mat[8], robot_t[2],
                0, 0, 0, 1);

        cv::Mat temp_rob_inv = temp_rob.inv();

        rob_rot_mat_inv[0] = temp_rob_inv.at<double>(0,0);
        rob_rot_mat_inv[1] = temp_rob_inv.at<double>(1,0);
        rob_rot_mat_inv[2] = temp_rob_inv.at<double>(2,0);
        rob_rot_mat_inv[3] = temp_rob_inv.at<double>(0,1);
        rob_rot_mat_inv[4] = temp_rob_inv.at<double>(1,1);
        rob_rot_mat_inv[5] = temp_rob_inv.at<double>(2,1);
        rob_rot_mat_inv[6] = temp_rob_inv.at<double>(0,2);
        rob_rot_mat_inv[7] = temp_rob_inv.at<double>(1,2);
        rob_rot_mat_inv[8] = temp_rob_inv.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rob_rot_mat_inv, rot_vec_rob);

        tras_vec_rob[0] = temp_rob_inv.at<double>(0,3);
        tras_vec_rob[1] = temp_rob_inv.at<double>(1,3);
        tras_vec_rob[2] = temp_rob_inv.at<double>(2,3);
        //###########################################################################################

        // Apply the robot position transformation
        ceres::AngleAxisRotatePoint(rot_vec_rob, tcp_pt, base_r);

        base_r[0] += tras_vec_rob[0];
        base_r[1] += tras_vec_rob[1];
        base_r[2] += tras_vec_rob[2];


        //###########################################################################################
        ceres::AngleAxisToRotationMatrix(b2w, cal_rot_mat);

        cv::Mat temp_cal = (cv::Mat_<double>(4,4) <<
                  cal_rot_mat[0], cal_rot_mat[3], cal_rot_mat[6], b2w[3],
                cal_rot_mat[1], cal_rot_mat[4], cal_rot_mat[7], b2w[4],
                cal_rot_mat[2], cal_rot_mat[5], cal_rot_mat[8], b2w[5],
                0, 0, 0, 1);

        cv::Mat temp_cal_inv = temp_cal.inv();

        cal_rot_mat_inv[0] = temp_cal_inv.at<double>(0,0);
        cal_rot_mat_inv[1] = temp_cal_inv.at<double>(1,0);
        cal_rot_mat_inv[2] = temp_cal_inv.at<double>(2,0);
        cal_rot_mat_inv[3] = temp_cal_inv.at<double>(0,1);
        cal_rot_mat_inv[4] = temp_cal_inv.at<double>(1,1);
        cal_rot_mat_inv[5] = temp_cal_inv.at<double>(2,1);
        cal_rot_mat_inv[6] = temp_cal_inv.at<double>(0,2);
        cal_rot_mat_inv[7] = temp_cal_inv.at<double>(1,2);
        cal_rot_mat_inv[8] = temp_cal_inv.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(cal_rot_mat_inv, rot_vec_cal);

        tras_vec_cal[0] = temp_cal_inv.at<double>(0,3);
        tras_vec_cal[1] = temp_cal_inv.at<double>(1,3);
        tras_vec_cal[2] = temp_cal_inv.at<double>(2,3);
        //###########################################################################################

        // Apply the robot position transformation
        ceres::AngleAxisRotatePoint(rot_vec_cal, base_r, cal_pt);

        cal_pt[0] += tras_vec_cal[0];
        cal_pt[1] += tras_vec_cal[1];
        cal_pt[2] += tras_vec_cal[2];

        //Chain2
        double pnp_r[3] = {double(pnp_r_vec(0)), double(pnp_r_vec(1)), double(pnp_r_vec(2))};
        double pnp_t[3] = {double(pnp_t_vec(0)), double(pnp_t_vec(1)), double(pnp_t_vec(2))};

        //###########################################################################################
        ceres::AngleAxisToRotationMatrix(pnp_r, cal_rot_mat2);

        cv::Mat temp_cal2 = (cv::Mat_<double>(4,4) <<
                cal_rot_mat2[0], cal_rot_mat2[3], cal_rot_mat2[6], pnp_t[0],
                cal_rot_mat2[1], cal_rot_mat2[4], cal_rot_mat2[7], pnp_t[1],
                cal_rot_mat2[2], cal_rot_mat2[5], cal_rot_mat2[8], pnp_t[2],
                0, 0, 0, 1);

        cv::Mat temp_cal_inv2 = temp_cal2.inv();

        cal_rot_mat_inv2[0] = temp_cal_inv2.at<double>(0,0);
        cal_rot_mat_inv2[1] = temp_cal_inv2.at<double>(1,0);
        cal_rot_mat_inv2[2] = temp_cal_inv2.at<double>(2,0);
        cal_rot_mat_inv2[3] = temp_cal_inv2.at<double>(0,1);
        cal_rot_mat_inv2[4] = temp_cal_inv2.at<double>(1,1);
        cal_rot_mat_inv2[5] = temp_cal_inv2.at<double>(2,1);
        cal_rot_mat_inv2[6] = temp_cal_inv2.at<double>(0,2);
        cal_rot_mat_inv2[7] = temp_cal_inv2.at<double>(1,2);
        cal_rot_mat_inv2[8] = temp_cal_inv2.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(cal_rot_mat_inv2, rot_vec_cal2);

        tras_vec_cal2[0] = temp_cal_inv2.at<double>(0,3);
        tras_vec_cal2[1] = temp_cal_inv2.at<double>(1,3);
        tras_vec_cal2[2] = temp_cal_inv2.at<double>(2,3);
        //###########################################################################################

        // Apply the current h2e transformation
        ceres::AngleAxisRotatePoint(rot_vec_cal2, cam_pt2, cal_pt2);

        cal_pt2[0] += tras_vec_cal2[0];
        cal_pt2[1] += tras_vec_cal2[1];
        cal_pt2[2] += tras_vec_cal2[2];

        residuals[0] = (cal_pt[0] - cal_pt2[0]);
        residuals[1] = (cal_pt[1] - cal_pt2[1]);
        residuals[2] = (cal_pt[2] - cal_pt2[2]);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const Eigen::Vector3d &pnp_r_vec,
                                        const Eigen::Vector3d &pnp_t_vec,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec,
                                        const Eigen::Vector3d &pattern_pt)
    {
        return new ceres::NumericDiffCostFunction<Classic_pnp_opt_inv, ceres::CENTRAL, 3, 6, 6, 6>(
                new Classic_pnp_opt_inv(pnp_r_vec, pnp_t_vec, robot_r_vec, robot_t_vec, pattern_pt));
    }

    Eigen::Vector3d pnp_r_vec;
    Eigen::Vector3d pnp_t_vec;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
    Eigen::Vector3d pattern_pt;
};


struct Classic_pnp_opt_new1
{
    Classic_pnp_opt_new1(const Eigen::Vector3d &pnp_r_vec,
                     const Eigen::Vector3d &pnp_t_vec,
                     const Eigen::Vector3d &robot_r_vec,
                     const Eigen::Vector3d &robot_t_vec) :
            pnp_r_vec(pnp_r_vec),
            pnp_t_vec(pnp_t_vec),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec){}

    //template <typename T>
    bool operator()(//const double* const weights,
            const double* const b2w,
            const double* const h2e,
            const double* const pnp_opt,
            double* residuals) const
    {
        double tcp_pt[3], tcp_pt_opt[3], tcp_pt2[3], base_r[3], cam_pt[3], cam_pt_pnp[3], rot_mat[9], rot_mat_inv_1[9], rot_mat_inv_2[9], rot_vec_1[3], tras_vec_1[3], rot_vec_2[3], tras_vec_2[3],
                rot_vec_rob[3], rot_vec_h2e[3], rot_vec_pnp[3], rot_mat_rob[9], rot_mat_h2e[9], rot_mat_pnp[9], rot_mat_opt[9], h2e_inv[3];


        double robot_r[3] = {double(robot_r_vec(0)), double(robot_r_vec(1)), double(robot_r_vec(2))},
                robot_t[3] = {double(robot_t_vec(0)), double(robot_t_vec(1)), double(robot_t_vec(2))};

        ceres::AngleAxisToRotationMatrix(robot_r, rot_mat_rob);
        cv::Mat A = (cv::Mat_<double>(4,4) <<
                rot_mat_rob[0], rot_mat_rob[3], rot_mat_rob[6], robot_t[0],
                rot_mat_rob[1], rot_mat_rob[4], rot_mat_rob[7], robot_t[1],
                rot_mat_rob[2], rot_mat_rob[5], rot_mat_rob[8], robot_t[2],
                0, 0, 0, 1);



        ceres::AngleAxisToRotationMatrix(h2e, rot_mat_h2e);
        cv::Mat X = (cv::Mat_<double>(4,4) <<
                rot_mat_h2e[0], rot_mat_h2e[3], rot_mat_h2e[6], h2e[3],
                rot_mat_h2e[1], rot_mat_h2e[4], rot_mat_h2e[7], h2e[4],
                rot_mat_h2e[2], rot_mat_h2e[5], rot_mat_h2e[8], h2e[5],
                0, 0, 0, 1);


        cv::Mat chain_1 = A*X; // T^{W}_{R}*T^{R}_{C}
        rot_mat_inv_1[0] = chain_1.at<double>(0,0);
        rot_mat_inv_1[1] = chain_1.at<double>(1,0);
        rot_mat_inv_1[2] = chain_1.at<double>(2,0);
        rot_mat_inv_1[3] = chain_1.at<double>(0,1);
        rot_mat_inv_1[4] = chain_1.at<double>(1,1);
        rot_mat_inv_1[5] = chain_1.at<double>(2,1);
        rot_mat_inv_1[6] = chain_1.at<double>(0,2);
        rot_mat_inv_1[7] = chain_1.at<double>(1,2);
        rot_mat_inv_1[8] = chain_1.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_1, rot_vec_1);

        tras_vec_1[0] = chain_1.at<double>(0,3);
        tras_vec_1[1] = chain_1.at<double>(1,3);
        tras_vec_1[2] = chain_1.at<double>(2,3);

        //Chain2
        //###########################################################################################
        ceres::AngleAxisToRotationMatrix(b2w, rot_mat);
        cv::Mat Z = (cv::Mat_<double>(4,4) <<
                rot_mat[0], rot_mat[3], rot_mat[6], b2w[3],
                rot_mat[1], rot_mat[4], rot_mat[7], b2w[4],
                rot_mat[2], rot_mat[5], rot_mat[8], b2w[5],
                0, 0, 0, 1);

        double pnp_r[3] = {double(pnp_r_vec(0)), double(pnp_r_vec(1)), double(pnp_r_vec(2))};
        double pnp_t[3] = {double(pnp_t_vec(0)), double(pnp_t_vec(1)), double(pnp_t_vec(2))};

        // Apply the current pnp transformation
        ceres::AngleAxisToRotationMatrix(pnp_r, rot_mat_pnp);
        cv::Mat B_1 = (cv::Mat_<double>(4,4) <<
                rot_mat_pnp[0], rot_mat_pnp[3], rot_mat_pnp[6], pnp_t[0],
                rot_mat_pnp[1], rot_mat_pnp[4], rot_mat_pnp[7], pnp_t[1],
                rot_mat_pnp[2], rot_mat_pnp[5], rot_mat_pnp[8], pnp_t[2],
                0, 0, 0, 1);

        cv::Mat B = B_1.inv();

        ceres::AngleAxisToRotationMatrix(pnp_opt, rot_mat_opt);
        cv::Mat W = (cv::Mat_<double>(4,4) <<
                rot_mat_opt[0], rot_mat_opt[3], rot_mat_opt[6], pnp_opt[3],
                rot_mat_opt[1], rot_mat_opt[4], rot_mat_opt[7], pnp_opt[4],
                rot_mat_opt[2], rot_mat_opt[5], rot_mat_opt[8], pnp_opt[5],
                0, 0, 0, 1);

        cv::Mat chain_2 = Z*B*W; // T^{W}_{B}*T^{B}_{C}*T^{C}_{C_opt}
        rot_mat_inv_2[0] = chain_2.at<double>(0,0);
        rot_mat_inv_2[1] = chain_2.at<double>(1,0);
        rot_mat_inv_2[2] = chain_2.at<double>(2,0);
        rot_mat_inv_2[3] = chain_2.at<double>(0,1);
        rot_mat_inv_2[4] = chain_2.at<double>(1,1);
        rot_mat_inv_2[5] = chain_2.at<double>(2,1);
        rot_mat_inv_2[6] = chain_2.at<double>(0,2);
        rot_mat_inv_2[7] = chain_2.at<double>(1,2);
        rot_mat_inv_2[8] = chain_2.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_2, rot_vec_2);

        tras_vec_2[0] = chain_2.at<double>(0,3);
        tras_vec_2[1] = chain_2.at<double>(1,3);
        tras_vec_2[2] = chain_2.at<double>(2,3);

        int lambda = 1;

        residuals[0] = (rot_vec_2[0] - rot_vec_1[0]) + lambda * pnp_opt[0];
        residuals[1] = (rot_vec_2[1] - rot_vec_1[1]) + lambda * pnp_opt[1];
        residuals[2] = (rot_vec_2[2] - rot_vec_1[2]) + lambda * pnp_opt[2];
        residuals[3] = (tras_vec_2[0] - tras_vec_1[0]) + lambda * pnp_opt[3];
        residuals[4] = (tras_vec_2[1] - tras_vec_1[1]) + lambda * pnp_opt[4];
        residuals[5] = (tras_vec_2[2] - tras_vec_1[2]) + lambda * pnp_opt[5];
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const Eigen::Vector3d &pnp_r_vec,
                                        const Eigen::Vector3d &pnp_t_vec,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec)
    {
        return new ceres::NumericDiffCostFunction<Classic_pnp_opt_new1, ceres::CENTRAL, 6, 6, 6, 6>(
                new Classic_pnp_opt_new1(pnp_r_vec, pnp_t_vec, robot_r_vec, robot_t_vec));
    }

    Eigen::Vector3d pnp_r_vec;
    Eigen::Vector3d pnp_t_vec;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
};

struct Classic_pnp_opt_new2
{
    Classic_pnp_opt_new2(const Eigen::Vector3d &pnp_r_vec,
                     const Eigen::Vector3d &pnp_t_vec,
                     const Eigen::Vector3d &robot_r_vec,
                     const Eigen::Vector3d &robot_t_vec) :
            pnp_r_vec(pnp_r_vec),
            pnp_t_vec(pnp_t_vec),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec){}

    bool operator()(
            const double* const b2w,
            const double* const h2e,
            const double* const pnp_opt,
            double* residuals) const
    {
        double tcp_pt[3], tcp_pt_opt[3], tcp_pt2[3], base_r[3], cam_pt[3], cam_pt_pnp[3], rot_mat[9], rot_mat_inv_1[9], rot_mat_inv_2[9], rot_vec_1[3], tras_vec_1[3], rot_vec_2[3], tras_vec_2[3],
                rot_vec_rob[3], rot_vec_h2e[3], rot_vec_pnp[3], rot_mat_rob[9], rot_mat_h2e[9], rot_mat_pnp[9], rot_mat_opt[9], h2e_inv[3];

        double robot_r[3] = {double(robot_r_vec(0)), double(robot_r_vec(1)), double(robot_r_vec(2))},
                robot_t[3] = {double(robot_t_vec(0)), double(robot_t_vec(1)), double(robot_t_vec(2))};

        ceres::AngleAxisToRotationMatrix(robot_r, rot_mat_rob);
        cv::Mat A = (cv::Mat_<double>(4,4) <<
                rot_mat_rob[0], rot_mat_rob[3], rot_mat_rob[6], robot_t[0],
                rot_mat_rob[1], rot_mat_rob[4], rot_mat_rob[7], robot_t[1],
                rot_mat_rob[2], rot_mat_rob[5], rot_mat_rob[8], robot_t[2],
                0, 0, 0, 1);



        ceres::AngleAxisToRotationMatrix(h2e, rot_mat_h2e);
        cv::Mat X = (cv::Mat_<double>(4,4) <<
                rot_mat_h2e[0], rot_mat_h2e[3], rot_mat_h2e[6], h2e[3],
                rot_mat_h2e[1], rot_mat_h2e[4], rot_mat_h2e[7], h2e[4],
                rot_mat_h2e[2], rot_mat_h2e[5], rot_mat_h2e[8], h2e[5],
                0, 0, 0, 1);



        //Chain2
        //###########################################################################################
        ceres::AngleAxisToRotationMatrix(b2w, rot_mat);
        cv::Mat Z = (cv::Mat_<double>(4,4) <<
                rot_mat[0], rot_mat[3], rot_mat[6], b2w[3],
                rot_mat[1], rot_mat[4], rot_mat[7], b2w[4],
                rot_mat[2], rot_mat[5], rot_mat[8], b2w[5],
                0, 0, 0, 1);


        //###########################################################################################

        double pnp_r[3] = {double(pnp_r_vec(0)), double(pnp_r_vec(1)), double(pnp_r_vec(2))};
        double pnp_t[3] = {double(pnp_t_vec(0)), double(pnp_t_vec(1)), double(pnp_t_vec(2))};

        // Apply the current pnp transformation
        ceres::AngleAxisToRotationMatrix(pnp_r, rot_mat_pnp);
        cv::Mat B_1 = (cv::Mat_<double>(4,4) <<
                rot_mat_pnp[0], rot_mat_pnp[3], rot_mat_pnp[6], pnp_t[0],
                rot_mat_pnp[1], rot_mat_pnp[4], rot_mat_pnp[7], pnp_t[1],
                rot_mat_pnp[2], rot_mat_pnp[5], rot_mat_pnp[8], pnp_t[2],
                0, 0, 0, 1);

        cv::Mat B = B_1.inv();

        ceres::AngleAxisToRotationMatrix(pnp_opt, rot_mat_opt);
        cv::Mat W = (cv::Mat_<double>(4,4) <<
                rot_mat_opt[0], rot_mat_opt[3], rot_mat_opt[6], pnp_opt[3],
                rot_mat_opt[1], rot_mat_opt[4], rot_mat_opt[7], pnp_opt[4],
                rot_mat_opt[2], rot_mat_opt[5], rot_mat_opt[8], pnp_opt[5],
                0, 0, 0, 1);

        cv::Mat chain_1 = Z.inv()*A; // T^{B}_{W} * T^{W}_{R}
        cv::Mat chain_2 = B*W*X.inv(); // T^{B}_{C} * T^{C}_{Ropt}

        rot_mat_inv_1[0] = chain_1.at<double>(0,0);
        rot_mat_inv_1[1] = chain_1.at<double>(1,0);
        rot_mat_inv_1[2] = chain_1.at<double>(2,0);
        rot_mat_inv_1[3] = chain_1.at<double>(0,1);
        rot_mat_inv_1[4] = chain_1.at<double>(1,1);
        rot_mat_inv_1[5] = chain_1.at<double>(2,1);
        rot_mat_inv_1[6] = chain_1.at<double>(0,2);
        rot_mat_inv_1[7] = chain_1.at<double>(1,2);
        rot_mat_inv_1[8] = chain_1.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_1, rot_vec_1);

        tras_vec_1[0] = chain_1.at<double>(0,3);
        tras_vec_1[1] = chain_1.at<double>(1,3);
        tras_vec_1[2] = chain_1.at<double>(2,3);

        rot_mat_inv_2[0] = chain_2.at<double>(0,0);
        rot_mat_inv_2[1] = chain_2.at<double>(1,0);
        rot_mat_inv_2[2] = chain_2.at<double>(2,0);
        rot_mat_inv_2[3] = chain_2.at<double>(0,1);
        rot_mat_inv_2[4] = chain_2.at<double>(1,1);
        rot_mat_inv_2[5] = chain_2.at<double>(2,1);
        rot_mat_inv_2[6] = chain_2.at<double>(0,2);
        rot_mat_inv_2[7] = chain_2.at<double>(1,2);
        rot_mat_inv_2[8] = chain_2.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_2, rot_vec_2);

        tras_vec_2[0] = chain_2.at<double>(0,3);
        tras_vec_2[1] = chain_2.at<double>(1,3);
        tras_vec_2[2] = chain_2.at<double>(2,3);

        int lambda = 1;

        residuals[0] = (rot_vec_2[0] - rot_vec_1[0]) + lambda * pnp_opt[0];
        residuals[1] = (rot_vec_2[1] - rot_vec_1[1]) + lambda * pnp_opt[1];
        residuals[2] = (rot_vec_2[2] - rot_vec_1[2]) + lambda * pnp_opt[2];
        residuals[3] = (tras_vec_2[0] - tras_vec_1[0]) + lambda * pnp_opt[3];
        residuals[4] = (tras_vec_2[1] - tras_vec_1[1]) + lambda * pnp_opt[4];
        residuals[5] = (tras_vec_2[2] - tras_vec_1[2]) + lambda * pnp_opt[5];
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const Eigen::Vector3d &pnp_r_vec,
                                        const Eigen::Vector3d &pnp_t_vec,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec)
    {
        return new ceres::NumericDiffCostFunction<Classic_pnp_opt_new2, ceres::CENTRAL, 6, 6, 6, 6>(
                new Classic_pnp_opt_new2(pnp_r_vec, pnp_t_vec, robot_r_vec, robot_t_vec));
    }

    Eigen::Vector3d pnp_r_vec;
    Eigen::Vector3d pnp_t_vec;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
};


struct Classic_pnp_opt_new3
{
    Classic_pnp_opt_new3(const Eigen::Vector3d &pnp_r_vec,
                         const Eigen::Vector3d &pnp_t_vec,
                         const Eigen::Vector3d &robot_r_vec,
                         const Eigen::Vector3d &robot_t_vec) :
            pnp_r_vec(pnp_r_vec),
            pnp_t_vec(pnp_t_vec),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec){}

    bool operator()(
            const double* const b2w,
            const double* const h2e,
            const double* const pnp_opt,
            double* residuals) const
    {
        double tcp_pt[3], tcp_pt_opt[3], tcp_pt2[3], base_r[3], cam_pt[3], cam_pt_pnp[3], rot_mat[9], rot_mat_inv_1[9], rot_mat_inv_2[9], rot_vec_1[3], tras_vec_1[3], rot_vec_2[3], tras_vec_2[3],
                rot_vec_rob[3], rot_vec_h2e[3], rot_vec_pnp[3], rot_mat_rob[9], rot_mat_h2e[9], rot_mat_pnp[9], rot_mat_opt[9], h2e_inv[3];

        double robot_r[3] = {double(robot_r_vec(0)), double(robot_r_vec(1)), double(robot_r_vec(2))},
                robot_t[3] = {double(robot_t_vec(0)), double(robot_t_vec(1)), double(robot_t_vec(2))};

        ceres::AngleAxisToRotationMatrix(robot_r, rot_mat_rob);
        cv::Mat A = (cv::Mat_<double>(4,4) <<
                                           rot_mat_rob[0], rot_mat_rob[3], rot_mat_rob[6], robot_t[0],
                rot_mat_rob[1], rot_mat_rob[4], rot_mat_rob[7], robot_t[1],
                rot_mat_rob[2], rot_mat_rob[5], rot_mat_rob[8], robot_t[2],
                0, 0, 0, 1);



        ceres::AngleAxisToRotationMatrix(h2e, rot_mat_h2e);
        cv::Mat X = (cv::Mat_<double>(4,4) <<
                                           rot_mat_h2e[0], rot_mat_h2e[3], rot_mat_h2e[6], h2e[3],
                rot_mat_h2e[1], rot_mat_h2e[4], rot_mat_h2e[7], h2e[4],
                rot_mat_h2e[2], rot_mat_h2e[5], rot_mat_h2e[8], h2e[5],
                0, 0, 0, 1);



        //Chain2
        //###########################################################################################
        ceres::AngleAxisToRotationMatrix(b2w, rot_mat);
        cv::Mat Z = (cv::Mat_<double>(4,4) <<
                                           rot_mat[0], rot_mat[3], rot_mat[6], b2w[3],
                rot_mat[1], rot_mat[4], rot_mat[7], b2w[4],
                rot_mat[2], rot_mat[5], rot_mat[8], b2w[5],
                0, 0, 0, 1);


        //###########################################################################################

        double pnp_r[3] = {double(pnp_r_vec(0)), double(pnp_r_vec(1)), double(pnp_r_vec(2))};
        double pnp_t[3] = {double(pnp_t_vec(0)), double(pnp_t_vec(1)), double(pnp_t_vec(2))};

        // Apply the current pnp transformation
        ceres::AngleAxisToRotationMatrix(pnp_r, rot_mat_pnp);
        cv::Mat B_1 = (cv::Mat_<double>(4,4) <<
                                             rot_mat_pnp[0], rot_mat_pnp[3], rot_mat_pnp[6], pnp_t[0],
                rot_mat_pnp[1], rot_mat_pnp[4], rot_mat_pnp[7], pnp_t[1],
                rot_mat_pnp[2], rot_mat_pnp[5], rot_mat_pnp[8], pnp_t[2],
                0, 0, 0, 1);

        cv::Mat B = B_1.inv();

        ceres::AngleAxisToRotationMatrix(pnp_opt, rot_mat_opt);
        cv::Mat W = (cv::Mat_<double>(4,4) <<
                                           rot_mat_opt[0], rot_mat_opt[3], rot_mat_opt[6], pnp_opt[3],
                rot_mat_opt[1], rot_mat_opt[4], rot_mat_opt[7], pnp_opt[4],
                rot_mat_opt[2], rot_mat_opt[5], rot_mat_opt[8], pnp_opt[5],
                0, 0, 0, 1);

        cv::Mat chain_1 = X*W.inv()*B.inv(); // T^{R}_{C}*T^{C_opt}_{C}*T^{C}_{B}
        cv::Mat chain_2 = A.inv()*Z; // T^{R}_{W}*T^{W}_{B}

        rot_mat_inv_1[0] = chain_1.at<double>(0,0);
        rot_mat_inv_1[1] = chain_1.at<double>(1,0);
        rot_mat_inv_1[2] = chain_1.at<double>(2,0);
        rot_mat_inv_1[3] = chain_1.at<double>(0,1);
        rot_mat_inv_1[4] = chain_1.at<double>(1,1);
        rot_mat_inv_1[5] = chain_1.at<double>(2,1);
        rot_mat_inv_1[6] = chain_1.at<double>(0,2);
        rot_mat_inv_1[7] = chain_1.at<double>(1,2);
        rot_mat_inv_1[8] = chain_1.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_1, rot_vec_1);

        tras_vec_1[0] = chain_1.at<double>(0,3);
        tras_vec_1[1] = chain_1.at<double>(1,3);
        tras_vec_1[2] = chain_1.at<double>(2,3);

        rot_mat_inv_2[0] = chain_2.at<double>(0,0);
        rot_mat_inv_2[1] = chain_2.at<double>(1,0);
        rot_mat_inv_2[2] = chain_2.at<double>(2,0);
        rot_mat_inv_2[3] = chain_2.at<double>(0,1);
        rot_mat_inv_2[4] = chain_2.at<double>(1,1);
        rot_mat_inv_2[5] = chain_2.at<double>(2,1);
        rot_mat_inv_2[6] = chain_2.at<double>(0,2);
        rot_mat_inv_2[7] = chain_2.at<double>(1,2);
        rot_mat_inv_2[8] = chain_2.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_2, rot_vec_2);

        tras_vec_2[0] = chain_2.at<double>(0,3);
        tras_vec_2[1] = chain_2.at<double>(1,3);
        tras_vec_2[2] = chain_2.at<double>(2,3);

        int lambda = 1;

        residuals[0] = (rot_vec_2[0] - rot_vec_1[0]) + lambda * pnp_opt[0];
        residuals[1] = (rot_vec_2[1] - rot_vec_1[1]) + lambda * pnp_opt[1];
        residuals[2] = (rot_vec_2[2] - rot_vec_1[2]) + lambda * pnp_opt[2];
        residuals[3] = (tras_vec_2[0] - tras_vec_1[0]) + lambda * pnp_opt[3];
        residuals[4] = (tras_vec_2[1] - tras_vec_1[1]) + lambda * pnp_opt[4];
        residuals[5] = (tras_vec_2[2] - tras_vec_1[2]) + lambda * pnp_opt[5];
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const Eigen::Vector3d &pnp_r_vec,
                                        const Eigen::Vector3d &pnp_t_vec,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec)
    {
        return new ceres::NumericDiffCostFunction<Classic_pnp_opt_new3, ceres::CENTRAL, 6, 6, 6, 6>(
                new Classic_pnp_opt_new3(pnp_r_vec, pnp_t_vec, robot_r_vec, robot_t_vec));
    }

    Eigen::Vector3d pnp_r_vec;
    Eigen::Vector3d pnp_t_vec;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
};

struct Classic_pnp_opt_new4
{
    Classic_pnp_opt_new4(const Eigen::Vector3d &pnp_r_vec,
                         const Eigen::Vector3d &pnp_t_vec,
                         const Eigen::Vector3d &robot_r_vec,
                         const Eigen::Vector3d &robot_t_vec) :
            pnp_r_vec(pnp_r_vec),
            pnp_t_vec(pnp_t_vec),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec){}

    bool operator()(
            const double* const b2w,
            const double* const h2e,
            const double* const pnp_opt,
            double* residuals) const
    {
        double tcp_pt[3], tcp_pt_opt[3], tcp_pt2[3], base_r[3], cam_pt[3], cam_pt_pnp[3], rot_mat[9], rot_mat_inv_1[9], rot_mat_inv_2[9], rot_vec_1[3], tras_vec_1[3], rot_vec_2[3], tras_vec_2[3],
                rot_vec_rob[3], rot_vec_h2e[3], rot_vec_pnp[3], rot_mat_rob[9], rot_mat_h2e[9], rot_mat_pnp[9], rot_mat_opt[9], h2e_inv[3];

        double robot_r[3] = {double(robot_r_vec(0)), double(robot_r_vec(1)), double(robot_r_vec(2))},
                robot_t[3] = {double(robot_t_vec(0)), double(robot_t_vec(1)), double(robot_t_vec(2))};

        ceres::AngleAxisToRotationMatrix(robot_r, rot_mat_rob);
        cv::Mat A = (cv::Mat_<double>(4,4) <<
                                           rot_mat_rob[0], rot_mat_rob[3], rot_mat_rob[6], robot_t[0],
                rot_mat_rob[1], rot_mat_rob[4], rot_mat_rob[7], robot_t[1],
                rot_mat_rob[2], rot_mat_rob[5], rot_mat_rob[8], robot_t[2],
                0, 0, 0, 1);



        ceres::AngleAxisToRotationMatrix(h2e, rot_mat_h2e);
        cv::Mat X = (cv::Mat_<double>(4,4) <<
                                           rot_mat_h2e[0], rot_mat_h2e[3], rot_mat_h2e[6], h2e[3],
                rot_mat_h2e[1], rot_mat_h2e[4], rot_mat_h2e[7], h2e[4],
                rot_mat_h2e[2], rot_mat_h2e[5], rot_mat_h2e[8], h2e[5],
                0, 0, 0, 1);



        //Chain2
        //###########################################################################################
        ceres::AngleAxisToRotationMatrix(b2w, rot_mat);
        cv::Mat Z = (cv::Mat_<double>(4,4) <<
                                           rot_mat[0], rot_mat[3], rot_mat[6], b2w[3],
                rot_mat[1], rot_mat[4], rot_mat[7], b2w[4],
                rot_mat[2], rot_mat[5], rot_mat[8], b2w[5],
                0, 0, 0, 1);


        //###########################################################################################

        double pnp_r[3] = {double(pnp_r_vec(0)), double(pnp_r_vec(1)), double(pnp_r_vec(2))};
        double pnp_t[3] = {double(pnp_t_vec(0)), double(pnp_t_vec(1)), double(pnp_t_vec(2))};

        // Apply the current pnp transformation
        ceres::AngleAxisToRotationMatrix(pnp_r, rot_mat_pnp);
        cv::Mat B_1 = (cv::Mat_<double>(4,4) <<
                                             rot_mat_pnp[0], rot_mat_pnp[3], rot_mat_pnp[6], pnp_t[0],
                rot_mat_pnp[1], rot_mat_pnp[4], rot_mat_pnp[7], pnp_t[1],
                rot_mat_pnp[2], rot_mat_pnp[5], rot_mat_pnp[8], pnp_t[2],
                0, 0, 0, 1);

        cv::Mat B = B_1.inv();

        ceres::AngleAxisToRotationMatrix(pnp_opt, rot_mat_opt);
        cv::Mat W = (cv::Mat_<double>(4,4) <<
                                           rot_mat_opt[0], rot_mat_opt[3], rot_mat_opt[6], pnp_opt[3],
                rot_mat_opt[1], rot_mat_opt[4], rot_mat_opt[7], pnp_opt[4],
                rot_mat_opt[2], rot_mat_opt[5], rot_mat_opt[8], pnp_opt[5],
                0, 0, 0, 1);

        cv::Mat chain_1 = W.inv()*B.inv()*Z.inv(); // T^{C_opt}_{C} * T^{C}_{B}*T^{B}_{W}
        cv::Mat chain_2 = X.inv()*A.inv(); // T^{C}_{R} * T^{R}_{W}

        rot_mat_inv_1[0] = chain_1.at<double>(0,0);
        rot_mat_inv_1[1] = chain_1.at<double>(1,0);
        rot_mat_inv_1[2] = chain_1.at<double>(2,0);
        rot_mat_inv_1[3] = chain_1.at<double>(0,1);
        rot_mat_inv_1[4] = chain_1.at<double>(1,1);
        rot_mat_inv_1[5] = chain_1.at<double>(2,1);
        rot_mat_inv_1[6] = chain_1.at<double>(0,2);
        rot_mat_inv_1[7] = chain_1.at<double>(1,2);
        rot_mat_inv_1[8] = chain_1.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_1, rot_vec_1);

        tras_vec_1[0] = chain_1.at<double>(0,3);
        tras_vec_1[1] = chain_1.at<double>(1,3);
        tras_vec_1[2] = chain_1.at<double>(2,3);

        rot_mat_inv_2[0] = chain_2.at<double>(0,0);
        rot_mat_inv_2[1] = chain_2.at<double>(1,0);
        rot_mat_inv_2[2] = chain_2.at<double>(2,0);
        rot_mat_inv_2[3] = chain_2.at<double>(0,1);
        rot_mat_inv_2[4] = chain_2.at<double>(1,1);
        rot_mat_inv_2[5] = chain_2.at<double>(2,1);
        rot_mat_inv_2[6] = chain_2.at<double>(0,2);
        rot_mat_inv_2[7] = chain_2.at<double>(1,2);
        rot_mat_inv_2[8] = chain_2.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_2, rot_vec_2);

        tras_vec_2[0] = chain_2.at<double>(0,3);
        tras_vec_2[1] = chain_2.at<double>(1,3);
        tras_vec_2[2] = chain_2.at<double>(2,3);

        int lambda = 1;

        residuals[0] = (rot_vec_2[0] - rot_vec_1[0]) + lambda * pnp_opt[0];
        residuals[1] = (rot_vec_2[1] - rot_vec_1[1]) + lambda * pnp_opt[1];
        residuals[2] = (rot_vec_2[2] - rot_vec_1[2]) + lambda * pnp_opt[2];
        residuals[3] = (tras_vec_2[0] - tras_vec_1[0]) + lambda * pnp_opt[3];
        residuals[4] = (tras_vec_2[1] - tras_vec_1[1]) + lambda * pnp_opt[4];
        residuals[5] = (tras_vec_2[2] - tras_vec_1[2]) + lambda * pnp_opt[5];
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const Eigen::Vector3d &pnp_r_vec,
                                        const Eigen::Vector3d &pnp_t_vec,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec)
    {
        return new ceres::NumericDiffCostFunction<Classic_pnp_opt_new4, ceres::CENTRAL, 6, 6, 6, 6>(
                new Classic_pnp_opt_new4(pnp_r_vec, pnp_t_vec, robot_r_vec, robot_t_vec));
    }

    Eigen::Vector3d pnp_r_vec;
    Eigen::Vector3d pnp_t_vec;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
};



struct Classic_pnp_opt1
{
    Classic_pnp_opt1(const Eigen::Vector3d &pnp_r_vec,
                     const Eigen::Vector3d &pnp_t_vec,
                     const Eigen::Vector3d &robot_r_vec,
                     const Eigen::Vector3d &robot_t_vec) :
            pnp_r_vec(pnp_r_vec),
            pnp_t_vec(pnp_t_vec),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec){}

    //template <typename T>
    bool operator()(//const double* const weights,
            const double* const b2w,
            const double* const h2e,
            const double* const robot_opt,
            double* residuals) const
    {
        double tcp_pt[3], tcp_pt_opt[3], tcp_pt2[3], base_r[3], cam_pt[3], cam_pt_pnp[3], rot_mat[9], rot_mat_inv_1[9], rot_mat_inv_2[9], rot_vec_1[3], tras_vec_1[3], rot_vec_2[3], tras_vec_2[3],
                rot_vec_rob[3], rot_vec_h2e[3], rot_vec_pnp[3], rot_mat_rob[9], rot_mat_h2e[9], rot_mat_pnp[9], rot_mat_opt[9], h2e_inv[3];


        double robot_r[3] = {double(robot_r_vec(0)), double(robot_r_vec(1)), double(robot_r_vec(2))},
                robot_t[3] = {double(robot_t_vec(0)), double(robot_t_vec(1)), double(robot_t_vec(2))};

        ceres::AngleAxisToRotationMatrix(robot_r, rot_mat_rob);
        cv::Mat A = (cv::Mat_<double>(4,4) <<
                rot_mat_rob[0], rot_mat_rob[3], rot_mat_rob[6], robot_t[0],
                rot_mat_rob[1], rot_mat_rob[4], rot_mat_rob[7], robot_t[1],
                rot_mat_rob[2], rot_mat_rob[5], rot_mat_rob[8], robot_t[2],
                0, 0, 0, 1);

        double robot_r_opt[3] = {double(0.0), double(0.0), double(robot_opt[0])},
                robot_t_opt[3] = {double(robot_opt[1]), double(robot_opt[2]), double(0.0)};

        ceres::AngleAxisToRotationMatrix(robot_r_opt, rot_mat_opt);
        cv::Mat temp_opt = (cv::Mat_<double>(4,4) <<
                rot_mat_opt[0], rot_mat_opt[3], rot_mat_opt[6], robot_t_opt[0],
                rot_mat_opt[1], rot_mat_opt[4], rot_mat_opt[7], robot_t_opt[1],
                rot_mat_opt[2], rot_mat_opt[5], rot_mat_opt[8], robot_t_opt[2],
                0, 0, 0, 1);

        ceres::AngleAxisToRotationMatrix(h2e, rot_mat_h2e);
        cv::Mat X = (cv::Mat_<double>(4,4) <<
                rot_mat_h2e[0], rot_mat_h2e[3], rot_mat_h2e[6], h2e[3],
                rot_mat_h2e[1], rot_mat_h2e[4], rot_mat_h2e[7], h2e[4],
                rot_mat_h2e[2], rot_mat_h2e[5], rot_mat_h2e[8], h2e[5],
                0, 0, 0, 1);


        //Chain2
        //###########################################################################################
        ceres::AngleAxisToRotationMatrix(b2w, rot_mat);
        cv::Mat Z = (cv::Mat_<double>(4,4) <<
                rot_mat[0], rot_mat[3], rot_mat[6], b2w[3],
                rot_mat[1], rot_mat[4], rot_mat[7], b2w[4],
                rot_mat[2], rot_mat[5], rot_mat[8], b2w[5],
                0, 0, 0, 1);


        //###########################################################################################

        double pnp_r[3] = {double(pnp_r_vec(0)), double(pnp_r_vec(1)), double(pnp_r_vec(2))};
        double pnp_t[3] = {double(pnp_t_vec(0)), double(pnp_t_vec(1)), double(pnp_t_vec(2))};

        // Apply the current pnp transformation
        ceres::AngleAxisToRotationMatrix(pnp_r, rot_mat_pnp);
        cv::Mat temp_pnp = (cv::Mat_<double>(4,4) <<
                rot_mat_pnp[0], rot_mat_pnp[3], rot_mat_pnp[6], pnp_t[0],
                rot_mat_pnp[1], rot_mat_pnp[4], rot_mat_pnp[7], pnp_t[1],
                rot_mat_pnp[2], rot_mat_pnp[5], rot_mat_pnp[8], pnp_t[2],
                0, 0, 0, 1);

        cv::Mat B = temp_pnp.inv();

        cv::Mat chain_1 = A*temp_opt*X;     // T^{R}_{C} * T^{C}_{B}
        cv::Mat chain_2 = Z*B;              // T^{Ropt}_{R} * T^{R}_{W} * T^{W}_{B}

        rot_mat_inv_1[0] = chain_1.at<double>(0,0);
        rot_mat_inv_1[1] = chain_1.at<double>(1,0);
        rot_mat_inv_1[2] = chain_1.at<double>(2,0);
        rot_mat_inv_1[3] = chain_1.at<double>(0,1);
        rot_mat_inv_1[4] = chain_1.at<double>(1,1);
        rot_mat_inv_1[5] = chain_1.at<double>(2,1);
        rot_mat_inv_1[6] = chain_1.at<double>(0,2);
        rot_mat_inv_1[7] = chain_1.at<double>(1,2);
        rot_mat_inv_1[8] = chain_1.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_1, rot_vec_1);

        tras_vec_1[0] = chain_1.at<double>(0,3);
        tras_vec_1[1] = chain_1.at<double>(1,3);
        tras_vec_1[2] = chain_1.at<double>(2,3);

        rot_mat_inv_2[0] = chain_2.at<double>(0,0);
        rot_mat_inv_2[1] = chain_2.at<double>(1,0);
        rot_mat_inv_2[2] = chain_2.at<double>(2,0);
        rot_mat_inv_2[3] = chain_2.at<double>(0,1);
        rot_mat_inv_2[4] = chain_2.at<double>(1,1);
        rot_mat_inv_2[5] = chain_2.at<double>(2,1);
        rot_mat_inv_2[6] = chain_2.at<double>(0,2);
        rot_mat_inv_2[7] = chain_2.at<double>(1,2);
        rot_mat_inv_2[8] = chain_2.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_2, rot_vec_2);

        tras_vec_2[0] = chain_2.at<double>(0,3);
        tras_vec_2[1] = chain_2.at<double>(1,3);
        tras_vec_2[2] = chain_2.at<double>(2,3);

        int lambda = 10;

        residuals[0] = (rot_vec_2[0] - rot_vec_1[0]);
        residuals[1] = (rot_vec_2[1] - rot_vec_1[1]);
        residuals[2] = (rot_vec_2[2] - rot_vec_1[2])   + lambda * robot_opt[0];
        residuals[3] = (tras_vec_2[0] - tras_vec_1[0]) + lambda * robot_opt[1];
        residuals[4] = (tras_vec_2[1] - tras_vec_1[1]) + lambda * robot_opt[2];
        residuals[5] = (tras_vec_2[2] - tras_vec_1[2]);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const Eigen::Vector3d &pnp_r_vec,
                                        const Eigen::Vector3d &pnp_t_vec,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec)
    {
        return new ceres::NumericDiffCostFunction<Classic_pnp_opt1, ceres::CENTRAL, 6, 6, 6, 3>(
                new Classic_pnp_opt1(pnp_r_vec, pnp_t_vec, robot_r_vec, robot_t_vec));
    }

    Eigen::Vector3d pnp_r_vec;
    Eigen::Vector3d pnp_t_vec;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
};



struct Classic_pnp_opt2
{
    Classic_pnp_opt2(const Eigen::Vector3d &pnp_r_vec,
                     const Eigen::Vector3d &pnp_t_vec,
                     const Eigen::Vector3d &robot_r_vec,
                     const Eigen::Vector3d &robot_t_vec) :
            pnp_r_vec(pnp_r_vec),
            pnp_t_vec(pnp_t_vec),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec){}

    //template <typename T>
    bool operator()(//const double* const weights,
            const double* const b2w,
            const double* const h2e,
            const double* const robot_opt,
            double* residuals) const
    {
        //T ptn_pt[3] = {T(pattern_pt(0)), T(pattern_pt(1)), T(pattern_pt(2))},
        //  tcp_pt[3], base_r[3], cam_pt[3];
        //double ptn_pt[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))},
        double tcp_pt[3], tcp_pt_opt[3], tcp_pt2[3], base_r[3], cam_pt[3], cam_pt_pnp[3], rot_mat[9], rot_mat_inv_1[9], rot_mat_inv_2[9], rot_vec_1[3], tras_vec_1[3], rot_vec_2[3], tras_vec_2[3],
                rot_vec_rob[3], rot_vec_h2e[3], rot_vec_pnp[3], rot_mat_rob[9], rot_mat_h2e[9], rot_mat_pnp[9], rot_mat_opt[9], h2e_inv[3];

        //double ptn_pt2[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))};


        double robot_r[3] = {double(robot_r_vec(0)), double(robot_r_vec(1)), double(robot_r_vec(2))},
                robot_t[3] = {double(robot_t_vec(0)), double(robot_t_vec(1)), double(robot_t_vec(2))};

        ceres::AngleAxisToRotationMatrix(robot_r, rot_mat_rob);
        cv::Mat temp_rob = (cv::Mat_<double>(4,4) <<
                                                  rot_mat_rob[0], rot_mat_rob[3], rot_mat_rob[6], robot_t[0],
                rot_mat_rob[1], rot_mat_rob[4], rot_mat_rob[7], robot_t[1],
                rot_mat_rob[2], rot_mat_rob[5], rot_mat_rob[8], robot_t[2],
                0, 0, 0, 1);

        ceres::AngleAxisToRotationMatrix(robot_opt, rot_mat_opt);
        cv::Mat temp_opt = (cv::Mat_<double>(4,4) <<
                                                  rot_mat_opt[0], rot_mat_opt[3], rot_mat_opt[6], robot_opt[3],
                rot_mat_opt[1], rot_mat_opt[4], rot_mat_opt[7], robot_opt[4],
                rot_mat_opt[2], rot_mat_opt[5], rot_mat_opt[8], robot_opt[5],
                0, 0, 0, 1);

        ceres::AngleAxisToRotationMatrix(h2e, rot_mat_h2e);
        cv::Mat temp_h2e = (cv::Mat_<double>(4,4) <<
                                                  rot_mat_h2e[0], rot_mat_h2e[3], rot_mat_h2e[6], h2e[3],
                rot_mat_h2e[1], rot_mat_h2e[4], rot_mat_h2e[7], h2e[4],
                rot_mat_h2e[2], rot_mat_h2e[5], rot_mat_h2e[8], h2e[5],
                0, 0, 0, 1);

        cv::Mat temp_h2e_inv = temp_h2e.inv();
        h2e_inv[0] = temp_h2e_inv.at<double>(0,3);
        h2e_inv[1] = temp_h2e_inv.at<double>(1,3);
        h2e_inv[2] = temp_h2e_inv.at<double>(2,3);



        //Chain2
        //###########################################################################################
        ceres::AngleAxisToRotationMatrix(b2w, rot_mat);
        cv::Mat temp_b2w = (cv::Mat_<double>(4,4) <<
                                                  rot_mat[0], rot_mat[3], rot_mat[6], b2w[3],
                rot_mat[1], rot_mat[4], rot_mat[7], b2w[4],
                rot_mat[2], rot_mat[5], rot_mat[8], b2w[5],
                0, 0, 0, 1);

        //cv::Mat temp_b2w_inv = temp_b2w.inv();

        //###########################################################################################

        double pnp_r[3] = {double(pnp_r_vec(0)), double(pnp_r_vec(1)), double(pnp_r_vec(2))};
        double pnp_t[3] = {double(pnp_t_vec(0)), double(pnp_t_vec(1)), double(pnp_t_vec(2))};

        // Apply the current pnp transformation
        ceres::AngleAxisToRotationMatrix(pnp_r, rot_mat_pnp);
        cv::Mat temp_pnp = (cv::Mat_<double>(4,4) <<
                                                  rot_mat_pnp[0], rot_mat_pnp[3], rot_mat_pnp[6], pnp_t[0],
                rot_mat_pnp[1], rot_mat_pnp[4], rot_mat_pnp[7], pnp_t[1],
                rot_mat_pnp[2], rot_mat_pnp[5], rot_mat_pnp[8], pnp_t[2],
                0, 0, 0, 1);

        //cv::Mat chain_1 = temp_h2e*temp_opt*temp_rob; // T^{C}_{EE} * T^{EE}_{E} * T^{E}_{W}
        //cv::Mat chain_2 = temp_pnp*temp_b2w.inv(); // T^{C}_{B} * T^{B}_{W}

        cv::Mat chain_1 = temp_b2w.inv()*temp_rob.inv()*temp_opt.inv(); // T^{B}_{W} * T^{W}_{R} * T^{R}_{Ropt}
        cv::Mat chain_2 = temp_pnp.inv()*temp_h2e; // T^{B}_{C} * T^{C}_{Ropt}

        rot_mat_inv_1[0] = chain_1.at<double>(0,0);
        rot_mat_inv_1[1] = chain_1.at<double>(1,0);
        rot_mat_inv_1[2] = chain_1.at<double>(2,0);
        rot_mat_inv_1[3] = chain_1.at<double>(0,1);
        rot_mat_inv_1[4] = chain_1.at<double>(1,1);
        rot_mat_inv_1[5] = chain_1.at<double>(2,1);
        rot_mat_inv_1[6] = chain_1.at<double>(0,2);
        rot_mat_inv_1[7] = chain_1.at<double>(1,2);
        rot_mat_inv_1[8] = chain_1.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_1, rot_vec_1);

        tras_vec_1[0] = chain_1.at<double>(0,3);
        tras_vec_1[1] = chain_1.at<double>(1,3);
        tras_vec_1[2] = chain_1.at<double>(2,3);

        rot_mat_inv_2[0] = chain_2.at<double>(0,0);
        rot_mat_inv_2[1] = chain_2.at<double>(1,0);
        rot_mat_inv_2[2] = chain_2.at<double>(2,0);
        rot_mat_inv_2[3] = chain_2.at<double>(0,1);
        rot_mat_inv_2[4] = chain_2.at<double>(1,1);
        rot_mat_inv_2[5] = chain_2.at<double>(2,1);
        rot_mat_inv_2[6] = chain_2.at<double>(0,2);
        rot_mat_inv_2[7] = chain_2.at<double>(1,2);
        rot_mat_inv_2[8] = chain_2.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_2, rot_vec_2);

        tras_vec_2[0] = chain_2.at<double>(0,3);
        tras_vec_2[1] = chain_2.at<double>(1,3);
        tras_vec_2[2] = chain_2.at<double>(2,3);

        int lambda = 10;

        residuals[0] = (rot_vec_2[0] - rot_vec_1[0]) + lambda * robot_opt[0];
        residuals[1] = (rot_vec_2[1] - rot_vec_1[1]) + lambda * robot_opt[1];
        residuals[2] = (rot_vec_2[2] - rot_vec_1[2]) + lambda * robot_opt[2];
        residuals[3] = (tras_vec_2[0] - tras_vec_1[0]) + lambda * robot_opt[3];
        residuals[4] = (tras_vec_2[1] - tras_vec_1[1]) + lambda * robot_opt[4];
        residuals[5] = (tras_vec_2[2] - tras_vec_1[2]) + lambda * robot_opt[5];
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const Eigen::Vector3d &pnp_r_vec,
                                        const Eigen::Vector3d &pnp_t_vec,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec)
    {
        return new ceres::NumericDiffCostFunction<Classic_pnp_opt2, ceres::CENTRAL, 6, 6, 6, 6>(
                new Classic_pnp_opt2(pnp_r_vec, pnp_t_vec, robot_r_vec, robot_t_vec));
    }

    Eigen::Vector3d pnp_r_vec;
    Eigen::Vector3d pnp_t_vec;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
};


struct Classic_pnp_opt3
{
    Classic_pnp_opt3(const Eigen::Vector3d &pnp_r_vec,
                          const Eigen::Vector3d &pnp_t_vec,
                          const Eigen::Vector3d &robot_r_vec,
                          const Eigen::Vector3d &robot_t_vec) :
            pnp_r_vec(pnp_r_vec),
            pnp_t_vec(pnp_t_vec),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec){}

    //template <typename T>
    bool operator()(//const double* const weights,
            const double* const b2w,
            const double* const h2e,
            const double* const robot_opt,
            double* residuals) const
    {
        //T ptn_pt[3] = {T(pattern_pt(0)), T(pattern_pt(1)), T(pattern_pt(2))},
        //  tcp_pt[3], base_r[3], cam_pt[3];
        //double ptn_pt[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))},
        double tcp_pt[3], tcp_pt_opt[3], tcp_pt2[3], base_r[3], cam_pt[3], cam_pt_pnp[3], rot_mat[9], rot_mat_inv_1[9], rot_mat_inv_2[9], rot_vec_1[3], tras_vec_1[3], rot_vec_2[3], tras_vec_2[3],
                rot_vec_rob[3], rot_vec_h2e[3], rot_vec_pnp[3], rot_mat_rob[9], rot_mat_h2e[9], rot_mat_pnp[9], rot_mat_opt[9], h2e_inv[3];

        //double ptn_pt2[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))};


        double robot_r[3] = {double(robot_r_vec(0)), double(robot_r_vec(1)), double(robot_r_vec(2))},
                robot_t[3] = {double(robot_t_vec(0)), double(robot_t_vec(1)), double(robot_t_vec(2))};

        ceres::AngleAxisToRotationMatrix(robot_r, rot_mat_rob);
        cv::Mat temp_rob = (cv::Mat_<double>(4,4) <<
                                                  rot_mat_rob[0], rot_mat_rob[3], rot_mat_rob[6], robot_t[0],
                rot_mat_rob[1], rot_mat_rob[4], rot_mat_rob[7], robot_t[1],
                rot_mat_rob[2], rot_mat_rob[5], rot_mat_rob[8], robot_t[2],
                0, 0, 0, 1);

        ceres::AngleAxisToRotationMatrix(robot_opt, rot_mat_opt);
        cv::Mat temp_opt = (cv::Mat_<double>(4,4) <<
                                                  rot_mat_opt[0], rot_mat_opt[3], rot_mat_opt[6], robot_opt[3],
                rot_mat_opt[1], rot_mat_opt[4], rot_mat_opt[7], robot_opt[4],
                rot_mat_opt[2], rot_mat_opt[5], rot_mat_opt[8], robot_opt[5],
                0, 0, 0, 1);

        ceres::AngleAxisToRotationMatrix(h2e, rot_mat_h2e);
        cv::Mat temp_h2e = (cv::Mat_<double>(4,4) <<
                                                  rot_mat_h2e[0], rot_mat_h2e[3], rot_mat_h2e[6], h2e[3],
                rot_mat_h2e[1], rot_mat_h2e[4], rot_mat_h2e[7], h2e[4],
                rot_mat_h2e[2], rot_mat_h2e[5], rot_mat_h2e[8], h2e[5],
                0, 0, 0, 1);

        cv::Mat temp_h2e_inv = temp_h2e.inv();
        h2e_inv[0] = temp_h2e_inv.at<double>(0,3);
        h2e_inv[1] = temp_h2e_inv.at<double>(1,3);
        h2e_inv[2] = temp_h2e_inv.at<double>(2,3);

        cv::Mat chain_1 = temp_h2e*temp_opt*temp_rob; // T^{C}_{EE} * T^{EE}_{E} * T^{E}_{W}
        rot_mat_inv_1[0] = chain_1.at<double>(0,0);
        rot_mat_inv_1[1] = chain_1.at<double>(1,0);
        rot_mat_inv_1[2] = chain_1.at<double>(2,0);
        rot_mat_inv_1[3] = chain_1.at<double>(0,1);
        rot_mat_inv_1[4] = chain_1.at<double>(1,1);
        rot_mat_inv_1[5] = chain_1.at<double>(2,1);
        rot_mat_inv_1[6] = chain_1.at<double>(0,2);
        rot_mat_inv_1[7] = chain_1.at<double>(1,2);
        rot_mat_inv_1[8] = chain_1.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_1, rot_vec_1);

        tras_vec_1[0] = chain_1.at<double>(0,3);
        tras_vec_1[1] = chain_1.at<double>(1,3);
        tras_vec_1[2] = chain_1.at<double>(2,3);

        //Chain2
        //###########################################################################################
        ceres::AngleAxisToRotationMatrix(b2w, rot_mat);
        cv::Mat temp_b2w = (cv::Mat_<double>(4,4) <<
                                                  rot_mat[0], rot_mat[3], rot_mat[6], b2w[3],
                rot_mat[1], rot_mat[4], rot_mat[7], b2w[4],
                rot_mat[2], rot_mat[5], rot_mat[8], b2w[5],
                0, 0, 0, 1);

        //cv::Mat temp_b2w_inv = temp_b2w.inv();

        //###########################################################################################

        double pnp_r[3] = {double(pnp_r_vec(0)), double(pnp_r_vec(1)), double(pnp_r_vec(2))};
        double pnp_t[3] = {double(pnp_t_vec(0)), double(pnp_t_vec(1)), double(pnp_t_vec(2))};

        // Apply the current pnp transformation
        ceres::AngleAxisToRotationMatrix(pnp_r, rot_mat_pnp);
        cv::Mat temp_pnp = (cv::Mat_<double>(4,4) <<
                                                  rot_mat_pnp[0], rot_mat_pnp[3], rot_mat_pnp[6], pnp_t[0],
                rot_mat_pnp[1], rot_mat_pnp[4], rot_mat_pnp[7], pnp_t[1],
                rot_mat_pnp[2], rot_mat_pnp[5], rot_mat_pnp[8], pnp_t[2],
                0, 0, 0, 1);

        cv::Mat chain_2 = temp_pnp*temp_b2w.inv(); // T^{C}_{B} * T^{B}_{W}
        rot_mat_inv_2[0] = chain_2.at<double>(0,0);
        rot_mat_inv_2[1] = chain_2.at<double>(1,0);
        rot_mat_inv_2[2] = chain_2.at<double>(2,0);
        rot_mat_inv_2[3] = chain_2.at<double>(0,1);
        rot_mat_inv_2[4] = chain_2.at<double>(1,1);
        rot_mat_inv_2[5] = chain_2.at<double>(2,1);
        rot_mat_inv_2[6] = chain_2.at<double>(0,2);
        rot_mat_inv_2[7] = chain_2.at<double>(1,2);
        rot_mat_inv_2[8] = chain_2.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_2, rot_vec_2);

        tras_vec_2[0] = chain_2.at<double>(0,3);
        tras_vec_2[1] = chain_2.at<double>(1,3);
        tras_vec_2[2] = chain_2.at<double>(2,3);

        int lambda = 10;

        residuals[0] = (rot_vec_2[0] - rot_vec_1[0]) + lambda * robot_opt[0];
        residuals[1] = (rot_vec_2[1] - rot_vec_1[1]) + lambda * robot_opt[1];
        residuals[2] = (rot_vec_2[2] - rot_vec_1[2]) + lambda * robot_opt[2];
        residuals[3] = (tras_vec_2[0] - tras_vec_1[0]) + lambda * robot_opt[3];
        residuals[4] = (tras_vec_2[1] - tras_vec_1[1]) + lambda * robot_opt[4];
        residuals[5] = (tras_vec_2[2] - tras_vec_1[2]) + lambda * robot_opt[5];
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const Eigen::Vector3d &pnp_r_vec,
                                        const Eigen::Vector3d &pnp_t_vec,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec)
    {
        return new ceres::NumericDiffCostFunction<Classic_pnp_opt3, ceres::CENTRAL, 6, 6, 6, 6>(
                new Classic_pnp_opt3(pnp_r_vec, pnp_t_vec, robot_r_vec, robot_t_vec));
    }

    Eigen::Vector3d pnp_r_vec;
    Eigen::Vector3d pnp_t_vec;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
};




struct Classic_pnp_opt4
{
    Classic_pnp_opt4(const Eigen::Vector3d &pnp_r_vec,
                          const Eigen::Vector3d &pnp_t_vec,
                          const Eigen::Vector3d &robot_r_vec,
                          const Eigen::Vector3d &robot_t_vec) :
            pnp_r_vec(pnp_r_vec),
            pnp_t_vec(pnp_t_vec),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec) {}

    //template <typename T>
    bool operator()(//const double* const weights,
            const double* const b2w,
            const double* const h2e,
            const double* const robot_opt,
            double* residuals) const
    {
        //T ptn_pt[3] = {T(pattern_pt(0)), T(pattern_pt(1)), T(pattern_pt(2))},
        //  tcp_pt[3], base_r[3], cam_pt[3];
        //double cam_pt[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))},
        double tcp_pt[3], tcp_pt_opt[3], cal_pt[3], base_r[3], base_pt[3], rot_mat_h2e[9], rot_mat_inv_h2e[9], rot_vec_h2e[3], tras_vec_h2e[3],
                opt_rot_mat[9], opt_rot_mat_inv[9], rot_vec_opt[3], tras_vec_opt[3],
                rob_rot_mat[9], rob_rot_mat_inv[9], rot_vec_rob[3], tras_vec_rob[3],
                cal_rot_mat[9], cal_rot_mat_inv[9], rot_vec_cal[3], tras_vec_cal[3],
                rot_mat_inv_1[9], rot_vec_1[3], tras_vec_1[3], rot_mat_inv_2[9],
                rot_vec_2[3], tras_vec_2[3], b2w_rot_mat[9], h2e_inv[3], b2w_inv[1];
        //double cam_pt2[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))};

        ceres::AngleAxisToRotationMatrix(h2e, rot_mat_h2e);
        cv::Mat temp_h2e = (cv::Mat_<double>(4,4) <<
                rot_mat_h2e[0], rot_mat_h2e[3], rot_mat_h2e[6], h2e[3],
                rot_mat_h2e[1], rot_mat_h2e[4], rot_mat_h2e[7], h2e[4],
                rot_mat_h2e[2], rot_mat_h2e[5], rot_mat_h2e[8], h2e[5],
                0, 0, 0, 1);

        cv::Mat temp_h2e_inv = temp_h2e.inv();

        ceres::AngleAxisToRotationMatrix(robot_opt, opt_rot_mat);
        cv::Mat temp_opt = (cv::Mat_<double>(4,4) <<
                opt_rot_mat[0], opt_rot_mat[3], opt_rot_mat[6], robot_opt[3],
                opt_rot_mat[1], opt_rot_mat[4], opt_rot_mat[7], robot_opt[4],
                opt_rot_mat[2], opt_rot_mat[5], opt_rot_mat[8], robot_opt[5],
                0, 0, 0, 1);

        cv::Mat temp_opt_inv = temp_opt.inv();

        h2e_inv[0] = temp_h2e_inv.at<double>(0,3);
        h2e_inv[1] = temp_h2e_inv.at<double>(1,3);
        h2e_inv[2] = temp_h2e_inv.at<double>(2,3);


        double robot_r[3] = {double(robot_r_vec(0)), double(robot_r_vec(1)), double(robot_r_vec(2))},
                robot_t[3] = {double(robot_t_vec(0)), double(robot_t_vec(1)), double(robot_t_vec(2))};

        ceres::AngleAxisToRotationMatrix(robot_r, rob_rot_mat);
        cv::Mat temp_rob = (cv::Mat_<double>(4,4) <<
                rob_rot_mat[0], rob_rot_mat[3], rob_rot_mat[6], robot_t[0],
                rob_rot_mat[1], rob_rot_mat[4], rob_rot_mat[7], robot_t[1],
                rob_rot_mat[2], rob_rot_mat[5], rob_rot_mat[8], robot_t[2],
                0, 0, 0, 1);

        cv::Mat temp_rob_inv = temp_rob.inv();

        cv::Mat chain1 = temp_rob_inv*temp_opt_inv*temp_h2e_inv; // T^{W}_{E} * T^{E}_{EE} * T^{EE}_{C}

        rot_mat_inv_1[0] = chain1.at<double>(0,0);
        rot_mat_inv_1[1] = chain1.at<double>(1,0);
        rot_mat_inv_1[2] = chain1.at<double>(2,0);
        rot_mat_inv_1[3] = chain1.at<double>(0,1);
        rot_mat_inv_1[4] = chain1.at<double>(1,1);
        rot_mat_inv_1[5] = chain1.at<double>(2,1);
        rot_mat_inv_1[6] = chain1.at<double>(0,2);
        rot_mat_inv_1[7] = chain1.at<double>(1,2);
        rot_mat_inv_1[8] = chain1.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_1, rot_vec_1);

        tras_vec_1[0] = chain1.at<double>(0,3);
        tras_vec_1[1] = chain1.at<double>(1,3);
        tras_vec_1[2] = chain1.at<double>(2,3);

        //###########################################################################################

        // Chain 2
        double pnp_r[3] = {double(pnp_r_vec(0)), double(pnp_r_vec(1)), double(pnp_r_vec(2))};
        double pnp_t[3] = {double(pnp_t_vec(0)), double(pnp_t_vec(1)), double(pnp_t_vec(2))};

        ceres::AngleAxisToRotationMatrix(pnp_r, cal_rot_mat);
        cv::Mat temp_cal = (cv::Mat_<double>(4,4) <<
                cal_rot_mat[0], cal_rot_mat[3], cal_rot_mat[6], pnp_t[0],
                cal_rot_mat[1], cal_rot_mat[4], cal_rot_mat[7], pnp_t[1],
                cal_rot_mat[2], cal_rot_mat[5], cal_rot_mat[8], pnp_t[2],
                0, 0, 0, 1);

        cv::Mat temp_cal_inv = temp_cal.inv();

        ceres::AngleAxisToRotationMatrix(b2w, b2w_rot_mat);
        cv::Mat temp_b2w = (cv::Mat_<double>(4,4) <<
                b2w_rot_mat[0], b2w_rot_mat[3], b2w_rot_mat[6], b2w[3],
                b2w_rot_mat[1], b2w_rot_mat[4], b2w_rot_mat[7], b2w[4],
                b2w_rot_mat[2], b2w_rot_mat[5], b2w_rot_mat[8], b2w[5],
                0, 0, 0, 1);

        cv::Mat chain2 = temp_b2w*temp_cal_inv; // T^{W}_{B} * T^{B}_{C}

        rot_mat_inv_2[0] = chain2.at<double>(0,0);
        rot_mat_inv_2[1] = chain2.at<double>(1,0);
        rot_mat_inv_2[2] = chain2.at<double>(2,0);
        rot_mat_inv_2[3] = chain2.at<double>(0,1);
        rot_mat_inv_2[4] = chain2.at<double>(1,1);
        rot_mat_inv_2[5] = chain2.at<double>(2,1);
        rot_mat_inv_2[6] = chain2.at<double>(0,2);
        rot_mat_inv_2[7] = chain2.at<double>(1,2);
        rot_mat_inv_2[8] = chain2.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_2, rot_vec_2);

        tras_vec_2[0] = chain2.at<double>(0,3);
        tras_vec_2[1] = chain2.at<double>(1,3);
        tras_vec_2[2] = chain2.at<double>(2,3);
        //###########################################################################################

        int lambda_robot = 10;
        double lambda_h2e = 0;

        residuals[0] = (rot_vec_2[0] - rot_vec_1[0]) + lambda_robot * robot_opt[0];
        residuals[1] = (rot_vec_2[1] - rot_vec_1[1]) + lambda_robot * robot_opt[1];
        residuals[2] = (rot_vec_2[2] - rot_vec_1[2]) + lambda_robot * robot_opt[2];
        residuals[3] = (tras_vec_2[0] - tras_vec_1[0]) + lambda_robot * robot_opt[3];
        residuals[4] = (tras_vec_2[1] - tras_vec_1[1]) + lambda_robot * robot_opt[4];
        residuals[5] = (tras_vec_2[2] - tras_vec_1[2]) + lambda_robot * robot_opt[5]; //+ 0.02 * h2e_inv[2]

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const Eigen::Vector3d &pnp_r_vec,
                                        const Eigen::Vector3d &pnp_t_vec,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec)
    {
        return new ceres::NumericDiffCostFunction<Classic_pnp_opt4, ceres::CENTRAL, 6, 6, 6, 6>(
                new Classic_pnp_opt4(pnp_r_vec, pnp_t_vec, robot_r_vec, robot_t_vec));
    }

    Eigen::Vector3d pnp_r_vec;
    Eigen::Vector3d pnp_t_vec;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
};






struct Classic_pnp_opt_test3
{
    Classic_pnp_opt_test3(const Eigen::Vector3d &pnp_r_vec,
                          const Eigen::Vector3d &pnp_t_vec,
                          const Eigen::Vector3d &robot_r_vec,
                          const Eigen::Vector3d &robot_t_vec) :
            pnp_r_vec(pnp_r_vec),
            pnp_t_vec(pnp_t_vec),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec){}

    //template <typename T>
    bool operator()(//const double* const weights,
            const double* const b2w,
            const double* const h2e,
            const double* const robot_opt,
            double* residuals) const
    {
        //T ptn_pt[3] = {T(pattern_pt(0)), T(pattern_pt(1)), T(pattern_pt(2))},
        //  tcp_pt[3], base_r[3], cam_pt[3];
        //double ptn_pt[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))},
        double tcp_pt[3], tcp_pt_opt[3], tcp_pt2[3], base_r[3], cam_pt[3], cam_pt_pnp[3], rot_mat[9], rot_mat_inv_1[9], rot_mat_inv_2[9], rot_vec_1[3], tras_vec_1[3], rot_vec_2[3], tras_vec_2[3],
                rot_vec_rob[3], rot_vec_h2e[3], rot_vec_pnp[3], rot_mat_rob[9], rot_mat_h2e[9], rot_mat_pnp[9], rot_mat_opt[9];
        //double ptn_pt2[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))};


        double robot_r[3] = {double(robot_r_vec(0)), double(robot_r_vec(1)), double(robot_r_vec(2))},
                robot_t[3] = {double(robot_t_vec(0)), double(robot_t_vec(1)), double(robot_t_vec(2))};

        ceres::AngleAxisToRotationMatrix(robot_r, rot_mat_rob);
        cv::Mat temp_rob = (cv::Mat_<double>(4,4) <<
                                                  rot_mat_rob[0], rot_mat_rob[3], rot_mat_rob[6], robot_t[0],
                rot_mat_rob[1], rot_mat_rob[4], rot_mat_rob[7], robot_t[1],
                rot_mat_rob[2], rot_mat_rob[5], rot_mat_rob[8], robot_t[2],
                0, 0, 0, 1);

        ceres::AngleAxisToRotationMatrix(robot_opt, rot_mat_opt);
        cv::Mat temp_opt = (cv::Mat_<double>(4,4) <<
                                                  rot_mat_opt[0], rot_mat_opt[3], rot_mat_opt[6], robot_opt[3],
                rot_mat_opt[1], rot_mat_opt[4], rot_mat_opt[7], robot_opt[4],
                rot_mat_opt[2], rot_mat_opt[5], rot_mat_opt[8], robot_opt[5],
                0, 0, 0, 1);

        ceres::AngleAxisToRotationMatrix(h2e, rot_mat_h2e);
        cv::Mat temp_h2e = (cv::Mat_<double>(4,4) <<
                                                  rot_mat_h2e[0], rot_mat_h2e[3], rot_mat_h2e[6], h2e[3],
                rot_mat_h2e[1], rot_mat_h2e[4], rot_mat_h2e[7], h2e[4],
                rot_mat_h2e[2], rot_mat_h2e[5], rot_mat_h2e[8], h2e[5],
                0, 0, 0, 1);

        cv::Mat chain_1 = temp_h2e*temp_rob*temp_opt;
        rot_mat_inv_1[0] = chain_1.at<double>(0,0);
        rot_mat_inv_1[1] = chain_1.at<double>(1,0);
        rot_mat_inv_1[2] = chain_1.at<double>(2,0);
        rot_mat_inv_1[3] = chain_1.at<double>(0,1);
        rot_mat_inv_1[4] = chain_1.at<double>(1,1);
        rot_mat_inv_1[5] = chain_1.at<double>(2,1);
        rot_mat_inv_1[6] = chain_1.at<double>(0,2);
        rot_mat_inv_1[7] = chain_1.at<double>(1,2);
        rot_mat_inv_1[8] = chain_1.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_1, rot_vec_1);

        tras_vec_1[0] = chain_1.at<double>(0,3);
        tras_vec_1[1] = chain_1.at<double>(1,3);
        tras_vec_1[2] = chain_1.at<double>(2,3);

        //Chain2
        //###########################################################################################
        ceres::AngleAxisToRotationMatrix(b2w, rot_mat);
        cv::Mat temp_b2w = (cv::Mat_<double>(4,4) <<
                                                  rot_mat[0], rot_mat[3], rot_mat[6], b2w[3],
                rot_mat[1], rot_mat[4], rot_mat[7], b2w[4],
                rot_mat[2], rot_mat[5], rot_mat[8], b2w[5],
                0, 0, 0, 1);

        cv::Mat temp_b2w_inv = temp_b2w.inv();

        //###########################################################################################

        double pnp_r[3] = {double(pnp_r_vec(0)), double(pnp_r_vec(1)), double(pnp_r_vec(2))};
        double pnp_t[3] = {double(pnp_t_vec(0)), double(pnp_t_vec(1)), double(pnp_t_vec(2))};

        // Apply the current pnp transformation
        ceres::AngleAxisToRotationMatrix(pnp_r, rot_mat_pnp);
        cv::Mat temp_pnp = (cv::Mat_<double>(4,4) <<
                                                  rot_mat_pnp[0], rot_mat_pnp[3], rot_mat_pnp[6], pnp_t[0],
                rot_mat_pnp[1], rot_mat_pnp[4], rot_mat_pnp[7], pnp_t[1],
                rot_mat_pnp[2], rot_mat_pnp[5], rot_mat_pnp[8], pnp_t[2],
                0, 0, 0, 1);

        cv::Mat chain_2 = temp_pnp*temp_b2w_inv;
        rot_mat_inv_2[0] = chain_2.at<double>(0,0);
        rot_mat_inv_2[1] = chain_2.at<double>(1,0);
        rot_mat_inv_2[2] = chain_2.at<double>(2,0);
        rot_mat_inv_2[3] = chain_2.at<double>(0,1);
        rot_mat_inv_2[4] = chain_2.at<double>(1,1);
        rot_mat_inv_2[5] = chain_2.at<double>(2,1);
        rot_mat_inv_2[6] = chain_2.at<double>(0,2);
        rot_mat_inv_2[7] = chain_2.at<double>(1,2);
        rot_mat_inv_2[8] = chain_2.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_2, rot_vec_2);

        tras_vec_2[0] = chain_2.at<double>(0,3);
        tras_vec_2[1] = chain_2.at<double>(1,3);
        tras_vec_2[2] = chain_2.at<double>(2,3);

        int lambda = 5;
        residuals[0] = (rot_vec_2[0] - rot_vec_1[0]) + lambda * robot_opt[0];
        residuals[1] = (rot_vec_2[1] - rot_vec_1[1]) + lambda * robot_opt[1];
        residuals[2] = (rot_vec_2[2] - rot_vec_1[2]) + lambda * robot_opt[2];
        residuals[3] = (tras_vec_2[0] - tras_vec_1[0]) + lambda * robot_opt[3];
        residuals[4] = (tras_vec_2[1] - tras_vec_1[1]) + lambda * robot_opt[4];
        residuals[5] = (tras_vec_2[2] - tras_vec_1[2]) + lambda * robot_opt[5];
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const Eigen::Vector3d &pnp_r_vec,
                                        const Eigen::Vector3d &pnp_t_vec,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec)
    {
        return new ceres::NumericDiffCostFunction<Classic_pnp_opt_test3, ceres::CENTRAL, 6, 6, 6, 6>(
                new Classic_pnp_opt_test3(pnp_r_vec, pnp_t_vec, robot_r_vec, robot_t_vec));
    }

    Eigen::Vector3d pnp_r_vec;
    Eigen::Vector3d pnp_t_vec;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
};


struct Classic_pnp_opt_test4
{
    Classic_pnp_opt_test4(const Eigen::Vector3d &pnp_r_vec,
                     const Eigen::Vector3d &pnp_t_vec,
                     const Eigen::Vector3d &robot_r_vec,
                     const Eigen::Vector3d &robot_t_vec) :
            pnp_r_vec(pnp_r_vec),
            pnp_t_vec(pnp_t_vec),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec) {}

    //template <typename T>
    bool operator()(//const double* const weights,
            const double* const b2w,
            const double* const h2e,
            const double* const robot_opt,
            double* residuals) const
    {
        //T ptn_pt[3] = {T(pattern_pt(0)), T(pattern_pt(1)), T(pattern_pt(2))},
        //  tcp_pt[3], base_r[3], cam_pt[3];
        //double cam_pt[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))},
        double tcp_pt[3], tcp_pt_opt[3], cal_pt[3], base_r[3], base_pt[3], rot_mat_h2e[9], rot_mat_inv_h2e[9], rot_vec_h2e[3], tras_vec_h2e[3],
                opt_rot_mat[9], opt_rot_mat_inv[9], rot_vec_opt[3], tras_vec_opt[3],
                rob_rot_mat[9], rob_rot_mat_inv[9], rot_vec_rob[3], tras_vec_rob[3],
                cal_rot_mat[9], cal_rot_mat_inv[9], rot_vec_cal[3], tras_vec_cal[3],
                rot_mat_inv_1[9], rot_vec_1[3], tras_vec_1[3], rot_mat_inv_2[9], rot_vec_2[3], tras_vec_2[3], b2w_rot_mat[9];
        //double cam_pt2[3] = {double(pattern_pt(0)), double(pattern_pt(1)), double(pattern_pt(2))};
        double quaternion_opt[4];

        ceres::AngleAxisToRotationMatrix(h2e, rot_mat_h2e);
        cv::Mat temp_h2e = (cv::Mat_<double>(4,4) <<
                rot_mat_h2e[0], rot_mat_h2e[3], rot_mat_h2e[6], h2e[3],
                rot_mat_h2e[1], rot_mat_h2e[4], rot_mat_h2e[7], h2e[4],
                rot_mat_h2e[2], rot_mat_h2e[5], rot_mat_h2e[8], h2e[5],
                0, 0, 0, 1);

        cv::Mat temp_h2e_inv = temp_h2e.inv();

        ceres::AngleAxisToRotationMatrix(robot_opt, opt_rot_mat);
        cv::Mat temp_opt = (cv::Mat_<double>(4,4) <<
                opt_rot_mat[0], opt_rot_mat[3], opt_rot_mat[6], robot_opt[3],
                opt_rot_mat[1], opt_rot_mat[4], opt_rot_mat[7], robot_opt[4],
                opt_rot_mat[2], opt_rot_mat[5], opt_rot_mat[8], robot_opt[5],
                0, 0, 0, 1);

        cv::Mat temp_opt_inv = temp_opt.inv();


        double robot_r[3] = {double(robot_r_vec(0)), double(robot_r_vec(1)), double(robot_r_vec(2))},
                robot_t[3] = {double(robot_t_vec(0)), double(robot_t_vec(1)), double(robot_t_vec(2))};

        ceres::AngleAxisToRotationMatrix(robot_r, rob_rot_mat);
        cv::Mat temp_rob = (cv::Mat_<double>(4,4) <<
                rob_rot_mat[0], rob_rot_mat[3], rob_rot_mat[6], robot_t[0],
                rob_rot_mat[1], rob_rot_mat[4], rob_rot_mat[7], robot_t[1],
                rob_rot_mat[2], rob_rot_mat[5], rob_rot_mat[8], robot_t[2],
                0, 0, 0, 1);

        cv::Mat temp_rob_inv = temp_rob.inv();

        cv::Mat chain1 = temp_opt_inv*temp_rob_inv*temp_h2e_inv;

        rot_mat_inv_1[0] = chain1.at<double>(0,0);
        rot_mat_inv_1[1] = chain1.at<double>(1,0);
        rot_mat_inv_1[2] = chain1.at<double>(2,0);
        rot_mat_inv_1[3] = chain1.at<double>(0,1);
        rot_mat_inv_1[4] = chain1.at<double>(1,1);
        rot_mat_inv_1[5] = chain1.at<double>(2,1);
        rot_mat_inv_1[6] = chain1.at<double>(0,2);
        rot_mat_inv_1[7] = chain1.at<double>(1,2);
        rot_mat_inv_1[8] = chain1.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_1, rot_vec_1);

        tras_vec_1[0] = chain1.at<double>(0,3);
        tras_vec_1[1] = chain1.at<double>(1,3);
        tras_vec_1[2] = chain1.at<double>(2,3);

        //###########################################################################################

        // Chain 2
        double pnp_r[3] = {double(pnp_r_vec(0)), double(pnp_r_vec(1)), double(pnp_r_vec(2))};
        double pnp_t[3] = {double(pnp_t_vec(0)), double(pnp_t_vec(1)), double(pnp_t_vec(2))};

        ceres::AngleAxisToRotationMatrix(pnp_r, cal_rot_mat);
        cv::Mat temp_cal = (cv::Mat_<double>(4,4) <<
                cal_rot_mat[0], cal_rot_mat[3], cal_rot_mat[6], pnp_t[0],
                cal_rot_mat[1], cal_rot_mat[4], cal_rot_mat[7], pnp_t[1],
                cal_rot_mat[2], cal_rot_mat[5], cal_rot_mat[8], pnp_t[2],
                0, 0, 0, 1);

        cv::Mat temp_cal_inv = temp_cal.inv();

        ceres::AngleAxisToRotationMatrix(b2w, b2w_rot_mat);
        cv::Mat temp_b2w = (cv::Mat_<double>(4,4) <<
                b2w_rot_mat[0], b2w_rot_mat[3], b2w_rot_mat[6], b2w[3],
                b2w_rot_mat[1], b2w_rot_mat[4], b2w_rot_mat[7], b2w[4],
                b2w_rot_mat[2], b2w_rot_mat[5], b2w_rot_mat[8], b2w[5],
                0, 0, 0, 1);

        cv::Mat chain2 = temp_b2w*temp_cal_inv;

        rot_mat_inv_2[0] = chain2.at<double>(0,0);
        rot_mat_inv_2[1] = chain2.at<double>(1,0);
        rot_mat_inv_2[2] = chain2.at<double>(2,0);
        rot_mat_inv_2[3] = chain2.at<double>(0,1);
        rot_mat_inv_2[4] = chain2.at<double>(1,1);
        rot_mat_inv_2[5] = chain2.at<double>(2,1);
        rot_mat_inv_2[6] = chain2.at<double>(0,2);
        rot_mat_inv_2[7] = chain2.at<double>(1,2);
        rot_mat_inv_2[8] = chain2.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_2, rot_vec_2);

        tras_vec_2[0] = chain2.at<double>(0,3);
        tras_vec_2[1] = chain2.at<double>(1,3);
        tras_vec_2[2] = chain2.at<double>(2,3);
        //###########################################################################################

        int lambda = 5;
        residuals[0] = (tras_vec_2[0] - tras_vec_1[0]) + lambda * robot_opt[3];
        residuals[1] = (tras_vec_2[1] - tras_vec_1[1]) + lambda * robot_opt[4];
        residuals[2] = (tras_vec_2[2] - tras_vec_1[2]) + lambda * robot_opt[5];
        residuals[3] = (rot_vec_2[0] - rot_vec_1[0]) + lambda * robot_opt[0];
        residuals[4] = (rot_vec_2[1] - rot_vec_1[1]) + lambda * robot_opt[1];
        residuals[5] = (rot_vec_2[2] - rot_vec_1[2]) + lambda * robot_opt[2];
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const Eigen::Vector3d &pnp_r_vec,
                                        const Eigen::Vector3d &pnp_t_vec,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec)
    {
        return new ceres::NumericDiffCostFunction<Classic_pnp_opt_test4, ceres::CENTRAL, 6, 6, 6, 6>(
                new Classic_pnp_opt_test4(pnp_r_vec, pnp_t_vec, robot_r_vec, robot_t_vec));
    }

    Eigen::Vector3d pnp_r_vec;
    Eigen::Vector3d pnp_t_vec;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
};


struct Classic_ax_xb_opt
{
    Classic_ax_xb_opt(const Eigen::Vector3d &pnp_r_vec_i_1,
          const Eigen::Vector3d &pnp_t_vec_i_1,
          const Eigen::Vector3d &robot_r_vec_i_1,
          const Eigen::Vector3d &robot_t_vec_i_1,
          const Eigen::Vector3d &pnp_r_vec_i,
          const Eigen::Vector3d &pnp_t_vec_i,
          const Eigen::Vector3d &robot_r_vec_i,
          const Eigen::Vector3d &robot_t_vec_i) :
            pnp_r_vec_i_1(pnp_r_vec_i_1),
            pnp_t_vec_i_1(pnp_t_vec_i_1),
            robot_r_vec_i_1(robot_r_vec_i_1),
            robot_t_vec_i_1(robot_t_vec_i_1),
            pnp_r_vec_i(pnp_r_vec_i),
            pnp_t_vec_i(pnp_t_vec_i),
            robot_r_vec_i(robot_r_vec_i),
            robot_t_vec_i(robot_t_vec_i){}

    bool operator()(const double* const h2e,
                    const double* const robot_opt_i_1,
                    const double* const robot_opt_i,
                    double* residuals) const
    {
        double tcp_pt[3], tcp_pt2[3], base_r[3], cam_pt_rob[3], cam_pt_pnp[3], cam_pt2_pnp[3], ee_pt2_rob[3], rot_mat[9], rot_mat_inv_1[9], rot_vec_1[3], tras_vec_1[3], rot_mat2[9], rot_mat_inv2[9], rot_vec_2[3], tras_vec_2[3],
                cal_rot_mat_i[9], cal_rot_mat_i_1[9], h2e_rot_mat[9], rot_mat_inv_2[9], rot_mat_rob_i_1[9], rot_mat_rob_i[9], opt_rot_mat_i_1[9], opt_rot_mat_i[9];

        ceres::AngleAxisToRotationMatrix(h2e, h2e_rot_mat);
        cv::Mat temp_h2e = (cv::Mat_<double>(4,4) <<
                h2e_rot_mat[0], h2e_rot_mat[3], h2e_rot_mat[6], h2e[3],
                h2e_rot_mat[1], h2e_rot_mat[4], h2e_rot_mat[7], h2e[4],
                h2e_rot_mat[2], h2e_rot_mat[5], h2e_rot_mat[8], h2e[5],
                0, 0, 0, 1);

        cv::Mat temp_h2e_inv = temp_h2e.inv();


        double robot_r_i_1[3] = {double(robot_r_vec_i_1(0)), double(robot_r_vec_i_1(1)), double(robot_r_vec_i_1(2))},
               robot_t_i_1[3] = {double(robot_t_vec_i_1(0)), double(robot_t_vec_i_1(1)), double(robot_t_vec_i_1(2))};

        double robot_r_i[3] = {double(robot_r_vec_i(0)), double(robot_r_vec_i(1)), double(robot_r_vec_i(2))},
               robot_t_i[3] = {double(robot_t_vec_i(0)), double(robot_t_vec_i(1)), double(robot_t_vec_i(2))};


        //###########################################################################################
        ceres::AngleAxisToRotationMatrix(robot_r_i_1, rot_mat_rob_i_1);

        cv::Mat temp_rob_i_1 = (cv::Mat_<double>(4,4) <<
                rot_mat_rob_i_1[0], rot_mat_rob_i_1[3], rot_mat_rob_i_1[6], robot_t_i_1[0],
                rot_mat_rob_i_1[1], rot_mat_rob_i_1[4], rot_mat_rob_i_1[7], robot_t_i_1[1],
                rot_mat_rob_i_1[2], rot_mat_rob_i_1[5], rot_mat_rob_i_1[8], robot_t_i_1[2],
                0, 0, 0, 1);

        cv::Mat temp_rob_i_1_inv = temp_rob_i_1.inv();

        ceres::AngleAxisToRotationMatrix(robot_opt_i_1, opt_rot_mat_i_1);
        cv::Mat temp_opt_i_1 = (cv::Mat_<double>(4,4) <<
                opt_rot_mat_i_1[0], opt_rot_mat_i_1[3], opt_rot_mat_i_1[6], robot_opt_i_1[3],
                opt_rot_mat_i_1[1], opt_rot_mat_i_1[4], opt_rot_mat_i_1[7], robot_opt_i_1[4],
                opt_rot_mat_i_1[2], opt_rot_mat_i_1[5], opt_rot_mat_i_1[8], robot_opt_i_1[5],
                0, 0, 0, 1);

        cv::Mat temp_opt_i_1_inv = temp_opt_i_1.inv();



        ceres::AngleAxisToRotationMatrix(robot_r_i, rot_mat_rob_i);

        cv::Mat temp_rob_i = (cv::Mat_<double>(4,4) <<
                rot_mat_rob_i[0], rot_mat_rob_i[3], rot_mat_rob_i[6], robot_t_i[0],
                rot_mat_rob_i[1], rot_mat_rob_i[4], rot_mat_rob_i[7], robot_t_i[1],
                rot_mat_rob_i[2], rot_mat_rob_i[5], rot_mat_rob_i[8], robot_t_i[2],
                0, 0, 0, 1);

        ceres::AngleAxisToRotationMatrix(robot_opt_i, opt_rot_mat_i);
        cv::Mat temp_opt_i = (cv::Mat_<double>(4,4) <<
                opt_rot_mat_i[0], opt_rot_mat_i[3], opt_rot_mat_i[6], robot_opt_i[3],
                opt_rot_mat_i[1], opt_rot_mat_i[4], opt_rot_mat_i[7], robot_opt_i[4],
                opt_rot_mat_i[2], opt_rot_mat_i[5], opt_rot_mat_i[8], robot_opt_i[5],
                0, 0, 0, 1);


        // T^{C}_{EE} * T^{EE}_{E} * T^{E}_{W}i-1*{W}_{E}i * T^{E}_{EE}
        cv::Mat chain1 = temp_h2e * temp_opt_i_1 * temp_rob_i_1 * temp_rob_i.inv() * temp_opt_i.inv();

        rot_mat_inv_1[0] = chain1.at<double>(0,0);
        rot_mat_inv_1[1] = chain1.at<double>(1,0);
        rot_mat_inv_1[2] = chain1.at<double>(2,0);
        rot_mat_inv_1[3] = chain1.at<double>(0,1);
        rot_mat_inv_1[4] = chain1.at<double>(1,1);
        rot_mat_inv_1[5] = chain1.at<double>(2,1);
        rot_mat_inv_1[6] = chain1.at<double>(0,2);
        rot_mat_inv_1[7] = chain1.at<double>(1,2);
        rot_mat_inv_1[8] = chain1.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_1, rot_vec_1);

        tras_vec_1[0] = chain1.at<double>(0,3);
        tras_vec_1[1] = chain1.at<double>(1,3);
        tras_vec_1[2] = chain1.at<double>(2,3);

        //###########################################################################################

        // Chain 2
        double pnp_r_i_1[3] = {double(pnp_r_vec_i_1(0)), double(pnp_r_vec_i_1(1)), double(pnp_r_vec_i_1(2))};
        double pnp_t_i_1[3] = {double(pnp_t_vec_i_1(0)), double(pnp_t_vec_i_1(1)), double(pnp_t_vec_i_1(2))};

        double pnp_r_i[3] = {double(pnp_r_vec_i(0)), double(pnp_r_vec_i(1)), double(pnp_r_vec_i(2))};
        double pnp_t_i[3] = {double(pnp_t_vec_i(0)), double(pnp_t_vec_i(1)), double(pnp_t_vec_i(2))};

        ceres::AngleAxisToRotationMatrix(pnp_r_i, cal_rot_mat_i);
        cv::Mat temp_cal_i = (cv::Mat_<double>(4,4) <<
                cal_rot_mat_i[0], cal_rot_mat_i[3], cal_rot_mat_i[6], pnp_t_i[0],
                cal_rot_mat_i[1], cal_rot_mat_i[4], cal_rot_mat_i[7], pnp_t_i[1],
                cal_rot_mat_i[2], cal_rot_mat_i[5], cal_rot_mat_i[8], pnp_t_i[2],
                0, 0, 0, 1);

        ceres::AngleAxisToRotationMatrix(pnp_r_i_1, cal_rot_mat_i_1);
        cv::Mat temp_cal_i_1 = (cv::Mat_<double>(4,4) <<
                cal_rot_mat_i_1[0], cal_rot_mat_i_1[3], cal_rot_mat_i_1[6], pnp_t_i_1[0],
                cal_rot_mat_i_1[1], cal_rot_mat_i_1[4], cal_rot_mat_i_1[7], pnp_t_i_1[1],
                cal_rot_mat_i_1[2], cal_rot_mat_i_1[5], cal_rot_mat_i_1[8], pnp_t_i_1[2],
                0, 0, 0, 1);

        cv::Mat temp_cal_i_1_inv = temp_cal_i_1.inv();

        // T^{C}_{B}i-1 * T^{B}_{C}i * T^{C}_{EE}
        cv::Mat chain2 = temp_cal_i_1 * temp_cal_i.inv() * temp_h2e;

        rot_mat_inv_2[0] = chain2.at<double>(0,0);
        rot_mat_inv_2[1] = chain2.at<double>(1,0);
        rot_mat_inv_2[2] = chain2.at<double>(2,0);
        rot_mat_inv_2[3] = chain2.at<double>(0,1);
        rot_mat_inv_2[4] = chain2.at<double>(1,1);
        rot_mat_inv_2[5] = chain2.at<double>(2,1);
        rot_mat_inv_2[6] = chain2.at<double>(0,2);
        rot_mat_inv_2[7] = chain2.at<double>(1,2);
        rot_mat_inv_2[8] = chain2.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_2, rot_vec_2);

        tras_vec_2[0] = chain2.at<double>(0,3);
        tras_vec_2[1] = chain2.at<double>(1,3);
        tras_vec_2[2] = chain2.at<double>(2,3);
        //###########################################################################################

        residuals[0] = (tras_vec_1[0] - tras_vec_2[0]);
        residuals[1] = (tras_vec_1[1] - tras_vec_2[1]);
        residuals[2] = (tras_vec_1[2] - tras_vec_2[2]);
        residuals[3] = (rot_vec_1[0] - rot_vec_2[0]);
        residuals[4] = (rot_vec_1[1] - rot_vec_2[1]);
        residuals[5] = (rot_vec_1[2] - rot_vec_2[2]);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const Eigen::Vector3d &pnp_r_vec_i_1,
                                        const Eigen::Vector3d &pnp_t_vec_i_1,
                                        const Eigen::Vector3d &robot_r_vec_i_1,
                                        const Eigen::Vector3d &robot_t_vec_i_1,
                                        const Eigen::Vector3d &pnp_r_vec_i,
                                        const Eigen::Vector3d &pnp_t_vec_i,
                                        const Eigen::Vector3d &robot_r_vec_i,
                                        const Eigen::Vector3d &robot_t_vec_i)
    {
        return new ceres::NumericDiffCostFunction<Classic_ax_xb_opt, ceres::CENTRAL, 6, 6, 6, 6>(
                new Classic_ax_xb_opt(pnp_r_vec_i_1, pnp_t_vec_i_1, robot_r_vec_i_1, robot_t_vec_i_1, pnp_r_vec_i, pnp_t_vec_i, robot_r_vec_i, robot_t_vec_i));
    }

    Eigen::Vector3d pnp_r_vec_i_1;
    Eigen::Vector3d pnp_t_vec_i_1;
    Eigen::Vector3d robot_r_vec_i_1;
    Eigen::Vector3d robot_t_vec_i_1;
    Eigen::Vector3d pnp_r_vec_i;
    Eigen::Vector3d pnp_t_vec_i;
    Eigen::Vector3d robot_r_vec_i;
    Eigen::Vector3d robot_t_vec_i;
};





struct Classic_ax_xb
{
    Classic_ax_xb(const Eigen::Vector3d &pnp_r_vec_i_1,
                      const Eigen::Vector3d &pnp_t_vec_i_1,
                      const Eigen::Vector3d &robot_r_vec_i_1,
                      const Eigen::Vector3d &robot_t_vec_i_1,
                      const Eigen::Vector3d &pnp_r_vec_i,
                      const Eigen::Vector3d &pnp_t_vec_i,
                      const Eigen::Vector3d &robot_r_vec_i,
                      const Eigen::Vector3d &robot_t_vec_i,
                      const double boardz_i,
                      const double camz_i) :
            pnp_r_vec_i_1(pnp_r_vec_i_1),
            pnp_t_vec_i_1(pnp_t_vec_i_1),
            robot_r_vec_i_1(robot_r_vec_i_1),
            robot_t_vec_i_1(robot_t_vec_i_1),
            pnp_r_vec_i(pnp_r_vec_i),
            pnp_t_vec_i(pnp_t_vec_i),
            robot_r_vec_i(robot_r_vec_i),
            robot_t_vec_i(robot_t_vec_i),
            boardz_i(boardz_i),
            camz_i(camz_i){}

    bool operator()(const double* const h2e,
                    double* residuals) const
    {
        double tcp_pt[3], tcp_pt2[3], base_r[3], cam_pt_rob[3], cam_pt_pnp[3], cam_pt2_pnp[3], ee_pt2_rob[3], rot_mat[9], rot_mat_inv_1[9], rot_vec_1[3], tras_vec_1[3], rot_mat2[9], rot_mat_inv2[9], rot_vec_2[3], tras_vec_2[3],
                cal_rot_mat_i[9], cal_rot_mat_i_1[9], h2e_rot_mat[9], rot_mat_inv_2[9], rot_mat_rob_i_1[9], rot_mat_rob_i[9], opt_rot_mat_i_1[9], opt_rot_mat_i[9], tras_vec_3[3], tras_vec_4[3];

        ceres::AngleAxisToRotationMatrix(h2e, h2e_rot_mat);
        cv::Mat X = (cv::Mat_<double>(4,4) <<
                                                  h2e_rot_mat[0], h2e_rot_mat[3], h2e_rot_mat[6], h2e[3],
                h2e_rot_mat[1], h2e_rot_mat[4], h2e_rot_mat[7], h2e[4],
                h2e_rot_mat[2], h2e_rot_mat[5], h2e_rot_mat[8], h2e[5],
                0, 0, 0, 1);


        double robot_r_i_1[3] = {double(robot_r_vec_i_1(0)), double(robot_r_vec_i_1(1)), double(robot_r_vec_i_1(2))},
                robot_t_i_1[3] = {double(robot_t_vec_i_1(0)), double(robot_t_vec_i_1(1)), double(robot_t_vec_i_1(2))};

        double robot_r_i[3] = {double(robot_r_vec_i(0)), double(robot_r_vec_i(1)), double(robot_r_vec_i(2))},
                robot_t_i[3] = {double(robot_t_vec_i(0)), double(robot_t_vec_i(1)), double(robot_t_vec_i(2))};


        //###########################################################################################
        ceres::AngleAxisToRotationMatrix(robot_r_i_1, rot_mat_rob_i_1);

        cv::Mat A_i_1 = (cv::Mat_<double>(4,4) <<
                rot_mat_rob_i_1[0], rot_mat_rob_i_1[3], rot_mat_rob_i_1[6], robot_t_i_1[0],
                rot_mat_rob_i_1[1], rot_mat_rob_i_1[4], rot_mat_rob_i_1[7], robot_t_i_1[1],
                rot_mat_rob_i_1[2], rot_mat_rob_i_1[5], rot_mat_rob_i_1[8], robot_t_i_1[2],
                0, 0, 0, 1);

        cv::Mat A_i_1_inv = A_i_1.inv();

        ceres::AngleAxisToRotationMatrix(robot_r_i, rot_mat_rob_i);

        cv::Mat A_i = (cv::Mat_<double>(4,4) <<
                rot_mat_rob_i[0], rot_mat_rob_i[3], rot_mat_rob_i[6], robot_t_i[0],
                rot_mat_rob_i[1], rot_mat_rob_i[4], rot_mat_rob_i[7], robot_t_i[1],
                rot_mat_rob_i[2], rot_mat_rob_i[5], rot_mat_rob_i[8], robot_t_i[2],
                0, 0, 0, 1);


        // T^{C}_{EE} * T^{EE}_{E} * T^{E}_{W}i-1*{W}_{E}i * T^{E}_{EE}
        cv::Mat chain1 = A_i_1_inv * A_i * X;
        cv::Mat chain3 = A_i *X;


        rot_mat_inv_1[0] = chain1.at<double>(0,0);
        rot_mat_inv_1[1] = chain1.at<double>(1,0);
        rot_mat_inv_1[2] = chain1.at<double>(2,0);
        rot_mat_inv_1[3] = chain1.at<double>(0,1);
        rot_mat_inv_1[4] = chain1.at<double>(1,1);
        rot_mat_inv_1[5] = chain1.at<double>(2,1);
        rot_mat_inv_1[6] = chain1.at<double>(0,2);
        rot_mat_inv_1[7] = chain1.at<double>(1,2);
        rot_mat_inv_1[8] = chain1.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_1, rot_vec_1);

        tras_vec_1[0] = chain1.at<double>(0,3);
        tras_vec_1[1] = chain1.at<double>(1,3);
        tras_vec_1[2] = chain1.at<double>(2,3);

        tras_vec_3[0] = chain3.at<double>(0,3);
        tras_vec_3[1] = chain3.at<double>(1,3);
        tras_vec_3[2] = chain3.at<double>(2,3);

        //###########################################################################################

        // Chain 2
        double pnp_r_i_1[3] = {double(pnp_r_vec_i_1(0)), double(pnp_r_vec_i_1(1)), double(pnp_r_vec_i_1(2))};
        double pnp_t_i_1[3] = {double(pnp_t_vec_i_1(0)), double(pnp_t_vec_i_1(1)), double(pnp_t_vec_i_1(2))};

        double pnp_r_i[3] = {double(pnp_r_vec_i(0)), double(pnp_r_vec_i(1)), double(pnp_r_vec_i(2))};
        double pnp_t_i[3] = {double(pnp_t_vec_i(0)), double(pnp_t_vec_i(1)), double(pnp_t_vec_i(2))};

        ceres::AngleAxisToRotationMatrix(pnp_r_i, cal_rot_mat_i);
        cv::Mat B_i = (cv::Mat_<double>(4,4) <<
                cal_rot_mat_i[0], cal_rot_mat_i[3], cal_rot_mat_i[6], pnp_t_i[0],
                cal_rot_mat_i[1], cal_rot_mat_i[4], cal_rot_mat_i[7], pnp_t_i[1],
                cal_rot_mat_i[2], cal_rot_mat_i[5], cal_rot_mat_i[8], pnp_t_i[2],
                0, 0, 0, 1);

        cv::Mat B_i_inv = B_i.inv();

        ceres::AngleAxisToRotationMatrix(pnp_r_i_1, cal_rot_mat_i_1);
        cv::Mat B_i_1 = (cv::Mat_<double>(4,4) <<
                cal_rot_mat_i_1[0], cal_rot_mat_i_1[3], cal_rot_mat_i_1[6], pnp_t_i_1[0],
                cal_rot_mat_i_1[1], cal_rot_mat_i_1[4], cal_rot_mat_i_1[7], pnp_t_i_1[1],
                cal_rot_mat_i_1[2], cal_rot_mat_i_1[5], cal_rot_mat_i_1[8], pnp_t_i_1[2],
                0, 0, 0, 1);



        // T^{C}_{B}i-1 * T^{B}_{C}i * T^{C}_{EE}
        cv::Mat chain2 = X * B_i_1 * B_i_inv;
        cv::Mat chain4 = X*B_i_1;

        rot_mat_inv_2[0] = chain2.at<double>(0,0);
        rot_mat_inv_2[1] = chain2.at<double>(1,0);
        rot_mat_inv_2[2] = chain2.at<double>(2,0);
        rot_mat_inv_2[3] = chain2.at<double>(0,1);
        rot_mat_inv_2[4] = chain2.at<double>(1,1);
        rot_mat_inv_2[5] = chain2.at<double>(2,1);
        rot_mat_inv_2[6] = chain2.at<double>(0,2);
        rot_mat_inv_2[7] = chain2.at<double>(1,2);
        rot_mat_inv_2[8] = chain2.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_2, rot_vec_2);

        tras_vec_2[0] = chain2.at<double>(0,3);
        tras_vec_2[1] = chain2.at<double>(1,3);
        tras_vec_2[2] = chain2.at<double>(2,3);

        tras_vec_4[0] = chain4.at<double>(0,3);
        tras_vec_4[1] = chain4.at<double>(1,3);
        tras_vec_4[2] = chain4.at<double>(2,3);
        //###########################################################################################

        residuals[0] = (tras_vec_1[0] - tras_vec_2[0]);
        residuals[1] = (tras_vec_1[1] - tras_vec_2[1]);
        residuals[2] = (tras_vec_1[2] - tras_vec_2[2]);
        residuals[3] = (rot_vec_1[0] - rot_vec_2[0]);
        residuals[4] = (rot_vec_1[1] - rot_vec_2[1]);
        residuals[5] = (rot_vec_1[2] - rot_vec_2[2]);
        residuals[6] = tras_vec_3[2] - abs(camz_i);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const Eigen::Vector3d &pnp_r_vec_i_1,
                                        const Eigen::Vector3d &pnp_t_vec_i_1,
                                        const Eigen::Vector3d &robot_r_vec_i_1,
                                        const Eigen::Vector3d &robot_t_vec_i_1,
                                        const Eigen::Vector3d &pnp_r_vec_i,
                                        const Eigen::Vector3d &pnp_t_vec_i,
                                        const Eigen::Vector3d &robot_r_vec_i,
                                        const Eigen::Vector3d &robot_t_vec_i,
                                        const double &boardz_i,
                                        const double &camz_i)
    {
        return new ceres::NumericDiffCostFunction<Classic_ax_xb, ceres::CENTRAL, 7, 6>(
                new Classic_ax_xb(pnp_r_vec_i_1, pnp_t_vec_i_1, robot_r_vec_i_1, robot_t_vec_i_1, pnp_r_vec_i, pnp_t_vec_i, robot_r_vec_i, robot_t_vec_i, boardz_i, camz_i));
    }

    Eigen::Vector3d pnp_r_vec_i_1;
    Eigen::Vector3d pnp_t_vec_i_1;
    Eigen::Vector3d robot_r_vec_i_1;
    Eigen::Vector3d robot_t_vec_i_1;
    Eigen::Vector3d pnp_r_vec_i;
    Eigen::Vector3d pnp_t_vec_i;
    Eigen::Vector3d robot_r_vec_i;
    Eigen::Vector3d robot_t_vec_i;
    double boardz_i;
    double camz_i;
};



struct Classic_ax_xb_rel
{
    Classic_ax_xb_rel(const Eigen::Vector3d &pnp_r_rel,
                  const Eigen::Vector3d &pnp_t_rel,
                  const Eigen::Vector3d &robot_r_rel,
                  const Eigen::Vector3d &robot_t_rel,
                  const Eigen::Vector3d &robot_r_vec,
                  const Eigen::Vector3d &robot_t_vec,
                  const double camz_i) :
            pnp_r_rel(pnp_r_rel),
            pnp_t_rel(pnp_t_rel),
            robot_r_rel(robot_r_rel),
            robot_t_rel(robot_t_rel),
            robot_r_vec(robot_r_vec),
            robot_t_vec(robot_t_vec),
            camz_i(camz_i){}

    bool operator()(const double* const h2e,
                    double* residuals) const
    {
        double tcp_pt[3], tcp_pt2[3], base_r[3], cam_pt_rob[3], cam_pt_pnp[3], cam_pt2_pnp[3], ee_pt2_rob[3], rot_mat[9], rot_mat_inv_1[9], rot_vec_1[3], tras_vec_1[3], rot_mat2[9], rot_mat_inv2[9], rot_vec_2[3], tras_vec_2[3],
                cal_rot_mat[9], h2e_rot_mat[9], rot_mat_inv_2[9], rot_mat_rob[9], rot_mat_rob_i[9], opt_rot_mat_i_1[9], opt_rot_mat_i[9], tras_vec_3[3], tras_vec_4[3];

        ceres::AngleAxisToRotationMatrix(h2e, h2e_rot_mat);
        cv::Mat X = (cv::Mat_<double>(4,4) <<
                h2e_rot_mat[0], h2e_rot_mat[3], h2e_rot_mat[6], h2e[3],
                h2e_rot_mat[1], h2e_rot_mat[4], h2e_rot_mat[7], h2e[4],
                h2e_rot_mat[2], h2e_rot_mat[5], h2e_rot_mat[8], h2e[5],
                0, 0, 0, 1);


        double robot_r[3] = {double(robot_r_rel(0)), double(robot_r_rel(1)), double(robot_r_rel(2))},
                robot_t[3] = {double(robot_t_rel(0)), double(robot_t_rel(1)), double(robot_t_rel(2))};

        double robot_r_i[3] = {double(robot_r_vec(0)), double(robot_r_vec(1)), double(robot_r_vec(2))},
                robot_t_i[3] = {double(robot_t_vec(0)), double(robot_t_vec(1)), double(robot_t_vec(2))};

        //###########################################################################################
        ceres::AngleAxisToRotationMatrix(robot_r, rot_mat_rob);

        cv::Mat A = (cv::Mat_<double>(4,4) <<
                rot_mat_rob[0], rot_mat_rob[3], rot_mat_rob[6], robot_t[0],
                rot_mat_rob[1], rot_mat_rob[4], rot_mat_rob[7], robot_t[1],
                rot_mat_rob[2], rot_mat_rob[5], rot_mat_rob[8], robot_t[2],
                0, 0, 0, 1);


        ceres::AngleAxisToRotationMatrix(robot_r_i, rot_mat_rob_i);

        cv::Mat Ai = (cv::Mat_<double>(4,4) <<
                rot_mat_rob_i[0], rot_mat_rob_i[3], rot_mat_rob_i[6], robot_t_i[0],
                rot_mat_rob_i[1], rot_mat_rob_i[4], rot_mat_rob_i[7], robot_t_i[1],
                rot_mat_rob_i[2], rot_mat_rob_i[5], rot_mat_rob_i[8], robot_t_i[2],
                0, 0, 0, 1);


        // T^{C}_{EE} * T^{EE}_{E} * T^{E}_{W}i-1*{W}_{E}i * T^{E}_{EE}
        cv::Mat chain1 = A* X;
        cv::Mat chain3 = X;


        rot_mat_inv_1[0] = chain1.at<double>(0,0);
        rot_mat_inv_1[1] = chain1.at<double>(1,0);
        rot_mat_inv_1[2] = chain1.at<double>(2,0);
        rot_mat_inv_1[3] = chain1.at<double>(0,1);
        rot_mat_inv_1[4] = chain1.at<double>(1,1);
        rot_mat_inv_1[5] = chain1.at<double>(2,1);
        rot_mat_inv_1[6] = chain1.at<double>(0,2);
        rot_mat_inv_1[7] = chain1.at<double>(1,2);
        rot_mat_inv_1[8] = chain1.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_1, rot_vec_1);

        tras_vec_1[0] = chain1.at<double>(0,3);
        tras_vec_1[1] = chain1.at<double>(1,3);
        tras_vec_1[2] = chain1.at<double>(2,3);

        tras_vec_3[0] = chain3.at<double>(0,3);
        tras_vec_3[1] = chain3.at<double>(1,3);
        tras_vec_3[2] = chain3.at<double>(2,3);

        //###########################################################################################

        // Chain 2
        double pnp_r[3] = {double(pnp_r_rel(0)), double(pnp_r_rel(1)), double(pnp_r_rel(2))};
        double pnp_t[3] = {double(pnp_t_rel(0)), double(pnp_t_rel(1)), double(pnp_t_rel(2))};

        ceres::AngleAxisToRotationMatrix(pnp_r, cal_rot_mat);
        cv::Mat B = (cv::Mat_<double>(4,4) <<
                cal_rot_mat[0], cal_rot_mat[3], cal_rot_mat[6], pnp_t[0],
                cal_rot_mat[1], cal_rot_mat[4], cal_rot_mat[7], pnp_t[1],
                cal_rot_mat[2], cal_rot_mat[5], cal_rot_mat[8], pnp_t[2],
                0, 0, 0, 1);


        // T^{C}_{B}i-1 * T^{B}_{C}i * T^{C}_{EE}
        cv::Mat chain2 = X * B;

        rot_mat_inv_2[0] = chain2.at<double>(0,0);
        rot_mat_inv_2[1] = chain2.at<double>(1,0);
        rot_mat_inv_2[2] = chain2.at<double>(2,0);
        rot_mat_inv_2[3] = chain2.at<double>(0,1);
        rot_mat_inv_2[4] = chain2.at<double>(1,1);
        rot_mat_inv_2[5] = chain2.at<double>(2,1);
        rot_mat_inv_2[6] = chain2.at<double>(0,2);
        rot_mat_inv_2[7] = chain2.at<double>(1,2);
        rot_mat_inv_2[8] = chain2.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_2, rot_vec_2);

        tras_vec_2[0] = chain2.at<double>(0,3);
        tras_vec_2[1] = chain2.at<double>(1,3);
        tras_vec_2[2] = chain2.at<double>(2,3);

        //###########################################################################################

        residuals[0] = (tras_vec_1[0] - tras_vec_2[0]);
        residuals[1] = (tras_vec_1[1] - tras_vec_2[1]);
        residuals[2] = (tras_vec_1[2] - tras_vec_2[2]);
        residuals[3] = (rot_vec_1[0] - rot_vec_2[0]);
        residuals[4] = (rot_vec_1[1] - rot_vec_2[1]);
        residuals[5] = (rot_vec_1[2] - rot_vec_2[2]);
        residuals[6] = tras_vec_3[2] - abs(camz_i);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const Eigen::Vector3d &pnp_r_rel,
                                        const Eigen::Vector3d &pnp_t_rel,
                                        const Eigen::Vector3d &robot_r_rel,
                                        const Eigen::Vector3d &robot_t_rel,
                                        const Eigen::Vector3d &robot_r_vec,
                                        const Eigen::Vector3d &robot_t_vec,
                                        const double &camz_i)
    {
        return new ceres::NumericDiffCostFunction<Classic_ax_xb_rel, ceres::CENTRAL, 7, 6>(
                new Classic_ax_xb_rel(pnp_r_rel, pnp_t_rel, robot_r_rel, robot_t_rel, robot_r_vec, robot_t_vec, camz_i));
    }

    Eigen::Vector3d pnp_r_rel;
    Eigen::Vector3d pnp_t_rel;
    Eigen::Vector3d robot_r_rel;
    Eigen::Vector3d robot_t_rel;
    Eigen::Vector3d robot_r_vec;
    Eigen::Vector3d robot_t_vec;
    double camz_i;
};




struct Classic_ax_xb_rel_multi
{
    Classic_ax_xb_rel_multi(const Eigen::Vector3d &pnp_r_rel1,
                      const Eigen::Vector3d &pnp_t_rel1,
                      const Eigen::Vector3d &pnp_r_rel2,
                      const Eigen::Vector3d &pnp_t_rel2) :
            pnp_r_rel1(pnp_r_rel1),
            pnp_t_rel1(pnp_t_rel1),
            pnp_r_rel2(pnp_r_rel2),
            pnp_t_rel2(pnp_t_rel2){}

    bool operator()(const double* const h2e1,
                    const double* const h2e2,
                    double* residuals) const
    {
        double  rot_mat[9], rot_mat_inv_1[9], rot_vec_1[3], tras_vec_1[3], rot_mat2[9], rot_mat_inv2[9], rot_vec_2[3], tras_vec_2[3], rot_mat_cam1[9], rot_mat_cam2[9],
                cal_rot_mat[9], h2e_rot_mat1[9], h2e_rot_mat2[9], rot_mat_inv_2[9], rot_mat_rob[9], tras_vec_3[3], tras_vec_4[3];

        ceres::AngleAxisToRotationMatrix(h2e1, h2e_rot_mat1);
        cv::Mat X1 = (cv::Mat_<double>(4,4) <<
                h2e_rot_mat1[0], h2e_rot_mat1[3], h2e_rot_mat1[6], h2e1[3],
                h2e_rot_mat1[1], h2e_rot_mat1[4], h2e_rot_mat1[7], h2e1[4],
                h2e_rot_mat1[2], h2e_rot_mat1[5], h2e_rot_mat1[8], h2e1[5],
                0, 0, 0, 1);

        ceres::AngleAxisToRotationMatrix(h2e2, h2e_rot_mat2);
        cv::Mat X2 = (cv::Mat_<double>(4,4) <<
                h2e_rot_mat2[0], h2e_rot_mat2[3], h2e_rot_mat2[6], h2e2[3],
                h2e_rot_mat2[1], h2e_rot_mat2[4], h2e_rot_mat2[7], h2e2[4],
                h2e_rot_mat2[2], h2e_rot_mat2[5], h2e_rot_mat2[8], h2e2[5],
                0, 0, 0, 1);


        double cam_r1[3] = {double(pnp_r_rel1(0)), double(pnp_r_rel1(1)), double(pnp_r_rel1(2))},
                cam_t1[3] = {double(pnp_t_rel1(0)), double(pnp_t_rel1(1)), double(pnp_t_rel1(2))};

        double cam_r2[3] = {double(pnp_r_rel2(0)), double(pnp_r_rel2(1)), double(pnp_r_rel2(2))},
                cam_t2[3] = {double(pnp_t_rel2(0)), double(pnp_t_rel2(1)), double(pnp_t_rel2(2))};

        ceres::AngleAxisToRotationMatrix(cam_r1, rot_mat_cam1);

        cv::Mat A = (cv::Mat_<double>(4,4) <<
                rot_mat_cam1[0], rot_mat_cam1[3], rot_mat_cam1[6], cam_t1[0],
                rot_mat_cam1[1], rot_mat_cam1[4], rot_mat_cam1[7], cam_t1[1],
                rot_mat_cam1[2], rot_mat_cam1[5], rot_mat_cam1[8], cam_t1[2],
                0, 0, 0, 1);


        ceres::AngleAxisToRotationMatrix(cam_r2, rot_mat_cam2);

        cv::Mat B = (cv::Mat_<double>(4,4) <<
                rot_mat_cam2[0], rot_mat_cam2[3], rot_mat_cam2[6], cam_t2[0],
                rot_mat_cam2[1], rot_mat_cam2[4], rot_mat_cam2[7], cam_t2[1],
                rot_mat_cam2[2], rot_mat_cam2[5], rot_mat_cam2[8], cam_t2[2],
                0, 0, 0, 1);


        // T^{C}_{EE} * T^{EE}_{E} * T^{E}_{W}i-1*{W}_{E}i * T^{E}_{EE}
        cv::Mat X = X1.inv()*X2;
        cv::Mat chain1 = A * X;
        cv::Mat chain2 = X * B;



        rot_mat_inv_1[0] = chain1.at<double>(0,0);
        rot_mat_inv_1[1] = chain1.at<double>(1,0);
        rot_mat_inv_1[2] = chain1.at<double>(2,0);
        rot_mat_inv_1[3] = chain1.at<double>(0,1);
        rot_mat_inv_1[4] = chain1.at<double>(1,1);
        rot_mat_inv_1[5] = chain1.at<double>(2,1);
        rot_mat_inv_1[6] = chain1.at<double>(0,2);
        rot_mat_inv_1[7] = chain1.at<double>(1,2);
        rot_mat_inv_1[8] = chain1.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_1, rot_vec_1);

        tras_vec_1[0] = chain1.at<double>(0,3);
        tras_vec_1[1] = chain1.at<double>(1,3);
        tras_vec_1[2] = chain1.at<double>(2,3);


        rot_mat_inv_2[0] = chain2.at<double>(0,0);
        rot_mat_inv_2[1] = chain2.at<double>(1,0);
        rot_mat_inv_2[2] = chain2.at<double>(2,0);
        rot_mat_inv_2[3] = chain2.at<double>(0,1);
        rot_mat_inv_2[4] = chain2.at<double>(1,1);
        rot_mat_inv_2[5] = chain2.at<double>(2,1);
        rot_mat_inv_2[6] = chain2.at<double>(0,2);
        rot_mat_inv_2[7] = chain2.at<double>(1,2);
        rot_mat_inv_2[8] = chain2.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_2, rot_vec_2);

        tras_vec_2[0] = chain2.at<double>(0,3);
        tras_vec_2[1] = chain2.at<double>(1,3);
        tras_vec_2[2] = chain2.at<double>(2,3);


        residuals[0] = (tras_vec_1[0] - tras_vec_2[0]);
        residuals[1] = (tras_vec_1[1] - tras_vec_2[1]);
        residuals[2] = (tras_vec_1[2] - tras_vec_2[2]);
        residuals[3] = (rot_vec_1[0] - rot_vec_2[0]);
        residuals[4] = (rot_vec_1[1] - rot_vec_2[1]);
        residuals[5] = (rot_vec_1[2] - rot_vec_2[2]);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const Eigen::Vector3d &pnp_r_rel1,
                                        const Eigen::Vector3d &pnp_t_rel1,
                                        const Eigen::Vector3d &pnp_r_rel2,
                                        const Eigen::Vector3d &pnp_t_rel2)
    {
        return new ceres::NumericDiffCostFunction<Classic_ax_xb_rel_multi, ceres::CENTRAL, 6, 6, 6>(
                new Classic_ax_xb_rel_multi(pnp_r_rel1, pnp_t_rel1, pnp_r_rel2, pnp_t_rel2));
    }

    Eigen::Vector3d pnp_r_rel1;
    Eigen::Vector3d pnp_t_rel1;
    Eigen::Vector3d pnp_r_rel2;
    Eigen::Vector3d pnp_t_rel2;
};




struct ax_xb_opt
{
    ax_xb_opt(const Eigen::Vector3d &pnp_r_vec_i_1,
                  const Eigen::Vector3d &pnp_t_vec_i_1,
                  const Eigen::Vector3d &robot_r_vec_i_1,
                  const Eigen::Vector3d &robot_t_vec_i_1,
                  const Eigen::Vector3d &pnp_r_vec_i,
                  const Eigen::Vector3d &pnp_t_vec_i,
                  const Eigen::Vector3d &robot_r_vec_i,
                  const Eigen::Vector3d &robot_t_vec_i,
                  const double camz_i) :
            pnp_r_vec_i_1(pnp_r_vec_i_1),
            pnp_t_vec_i_1(pnp_t_vec_i_1),
            robot_r_vec_i_1(robot_r_vec_i_1),
            robot_t_vec_i_1(robot_t_vec_i_1),
            pnp_r_vec_i(pnp_r_vec_i),
            pnp_t_vec_i(pnp_t_vec_i),
            robot_r_vec_i(robot_r_vec_i),
            robot_t_vec_i(robot_t_vec_i),
            camz_i(camz_i){}

    bool operator()(const double* const h2e,
                    const double* const robot_opt,
                    double* residuals) const
    {
        double tcp_pt[3], tcp_pt2[3], base_r[3], cam_pt_rob[3], cam_pt_pnp[3], cam_pt2_pnp[3], ee_pt2_rob[3], rot_mat[9], rot_mat_inv_1[9], rot_vec_1[3], tras_vec_1[3], rot_mat2[9], rot_mat_inv2[9], rot_vec_2[3], tras_vec_2[3],
                cal_rot_mat_i[9], cal_rot_mat_i_1[9], h2e_rot_mat[9], rot_mat_inv_2[9], rot_mat_rob_i_1[9], rot_mat_rob_i[9], opt_rot_mat_i_1[9], opt_rot_mat_i[9], tras_vec_3[3], tras_vec_4[3], rot_mat_opt[9];

        ceres::AngleAxisToRotationMatrix(h2e, h2e_rot_mat);
        cv::Mat X = (cv::Mat_<double>(4,4) <<
                                           h2e_rot_mat[0], h2e_rot_mat[3], h2e_rot_mat[6], h2e[3],
                h2e_rot_mat[1], h2e_rot_mat[4], h2e_rot_mat[7], h2e[4],
                h2e_rot_mat[2], h2e_rot_mat[5], h2e_rot_mat[8], h2e[5],
                0, 0, 0, 1);


        double robot_r_i_1[3] = {double(robot_r_vec_i_1(0)), double(robot_r_vec_i_1(1)), double(robot_r_vec_i_1(2))},
                robot_t_i_1[3] = {double(robot_t_vec_i_1(0)), double(robot_t_vec_i_1(1)), double(robot_t_vec_i_1(2))};

        double robot_r_i[3] = {double(robot_r_vec_i(0)), double(robot_r_vec_i(1)), double(robot_r_vec_i(2))},
                robot_t_i[3] = {double(robot_t_vec_i(0)), double(robot_t_vec_i(1)), double(robot_t_vec_i(2))};


        //###########################################################################################
        ceres::AngleAxisToRotationMatrix(robot_r_i_1, rot_mat_rob_i_1);

        cv::Mat A_i_1 = (cv::Mat_<double>(4,4) <<
                                               rot_mat_rob_i_1[0], rot_mat_rob_i_1[3], rot_mat_rob_i_1[6], robot_t_i_1[0],
                rot_mat_rob_i_1[1], rot_mat_rob_i_1[4], rot_mat_rob_i_1[7], robot_t_i_1[1],
                rot_mat_rob_i_1[2], rot_mat_rob_i_1[5], rot_mat_rob_i_1[8], robot_t_i_1[2],
                0, 0, 0, 1);

        cv::Mat A_i_1_inv = A_i_1.inv();

        ceres::AngleAxisToRotationMatrix(robot_r_i, rot_mat_rob_i);

        cv::Mat A_i = (cv::Mat_<double>(4,4) <<
                                             rot_mat_rob_i[0], rot_mat_rob_i[3], rot_mat_rob_i[6], robot_t_i[0],
                rot_mat_rob_i[1], rot_mat_rob_i[4], rot_mat_rob_i[7], robot_t_i[1],
                rot_mat_rob_i[2], rot_mat_rob_i[5], rot_mat_rob_i[8], robot_t_i[2],
                0, 0, 0, 1);

        double robot_r_opt[3] = {double(0.0), double(0.0), double(robot_opt[0])},
                robot_t_opt[3] = {double(robot_opt[1]), double(robot_opt[2]), double(0.0)};

        ceres::AngleAxisToRotationMatrix(robot_r_opt, rot_mat_opt);
        cv::Mat W = (cv::Mat_<double>(4,4) <<
                rot_mat_opt[0], rot_mat_opt[3], rot_mat_opt[6], robot_t_opt[0],
                rot_mat_opt[1], rot_mat_opt[4], rot_mat_opt[7], robot_t_opt[1],
                rot_mat_opt[2], rot_mat_opt[5], rot_mat_opt[8], robot_t_opt[2],
                0, 0, 0, 1);


        // T^{C}_{EE} * T^{EE}_{E} * T^{E}_{W}i-1*{W}_{E}i * T^{E}_{EE}
        cv::Mat chain1 = A_i_1_inv * A_i * W * X;
        cv::Mat chain3 = A_i *X;


        rot_mat_inv_1[0] = chain1.at<double>(0,0);
        rot_mat_inv_1[1] = chain1.at<double>(1,0);
        rot_mat_inv_1[2] = chain1.at<double>(2,0);
        rot_mat_inv_1[3] = chain1.at<double>(0,1);
        rot_mat_inv_1[4] = chain1.at<double>(1,1);
        rot_mat_inv_1[5] = chain1.at<double>(2,1);
        rot_mat_inv_1[6] = chain1.at<double>(0,2);
        rot_mat_inv_1[7] = chain1.at<double>(1,2);
        rot_mat_inv_1[8] = chain1.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_1, rot_vec_1);

        tras_vec_1[0] = chain1.at<double>(0,3);
        tras_vec_1[1] = chain1.at<double>(1,3);
        tras_vec_1[2] = chain1.at<double>(2,3);

        tras_vec_3[0] = chain3.at<double>(0,3);
        tras_vec_3[1] = chain3.at<double>(1,3);
        tras_vec_3[2] = chain3.at<double>(2,3);

        //###########################################################################################

        // Chain 2
        double pnp_r_i_1[3] = {double(pnp_r_vec_i_1(0)), double(pnp_r_vec_i_1(1)), double(pnp_r_vec_i_1(2))};
        double pnp_t_i_1[3] = {double(pnp_t_vec_i_1(0)), double(pnp_t_vec_i_1(1)), double(pnp_t_vec_i_1(2))};

        double pnp_r_i[3] = {double(pnp_r_vec_i(0)), double(pnp_r_vec_i(1)), double(pnp_r_vec_i(2))};
        double pnp_t_i[3] = {double(pnp_t_vec_i(0)), double(pnp_t_vec_i(1)), double(pnp_t_vec_i(2))};

        ceres::AngleAxisToRotationMatrix(pnp_r_i, cal_rot_mat_i);
        cv::Mat B_i = (cv::Mat_<double>(4,4) <<
                                             cal_rot_mat_i[0], cal_rot_mat_i[3], cal_rot_mat_i[6], pnp_t_i[0],
                cal_rot_mat_i[1], cal_rot_mat_i[4], cal_rot_mat_i[7], pnp_t_i[1],
                cal_rot_mat_i[2], cal_rot_mat_i[5], cal_rot_mat_i[8], pnp_t_i[2],
                0, 0, 0, 1);

        cv::Mat B_i_inv = B_i.inv();

        ceres::AngleAxisToRotationMatrix(pnp_r_i_1, cal_rot_mat_i_1);
        cv::Mat B_i_1 = (cv::Mat_<double>(4,4) <<
                                               cal_rot_mat_i_1[0], cal_rot_mat_i_1[3], cal_rot_mat_i_1[6], pnp_t_i_1[0],
                cal_rot_mat_i_1[1], cal_rot_mat_i_1[4], cal_rot_mat_i_1[7], pnp_t_i_1[1],
                cal_rot_mat_i_1[2], cal_rot_mat_i_1[5], cal_rot_mat_i_1[8], pnp_t_i_1[2],
                0, 0, 0, 1);



        // T^{C}_{B}i-1 * T^{B}_{C}i * T^{C}_{EE}
        cv::Mat chain2 = X * B_i_1 * B_i_inv;
        cv::Mat chain4 = X*B_i_1;

        rot_mat_inv_2[0] = chain2.at<double>(0,0);
        rot_mat_inv_2[1] = chain2.at<double>(1,0);
        rot_mat_inv_2[2] = chain2.at<double>(2,0);
        rot_mat_inv_2[3] = chain2.at<double>(0,1);
        rot_mat_inv_2[4] = chain2.at<double>(1,1);
        rot_mat_inv_2[5] = chain2.at<double>(2,1);
        rot_mat_inv_2[6] = chain2.at<double>(0,2);
        rot_mat_inv_2[7] = chain2.at<double>(1,2);
        rot_mat_inv_2[8] = chain2.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_2, rot_vec_2);

        tras_vec_2[0] = chain2.at<double>(0,3);
        tras_vec_2[1] = chain2.at<double>(1,3);
        tras_vec_2[2] = chain2.at<double>(2,3);

        tras_vec_4[0] = chain4.at<double>(0,3);
        tras_vec_4[1] = chain4.at<double>(1,3);
        tras_vec_4[2] = chain4.at<double>(2,3);
        //###########################################################################################

        int lambda = 5;
        residuals[0] = (tras_vec_1[0] - tras_vec_2[0]);//+ lambda * robot_opt[1];
        residuals[1] = (tras_vec_1[1] - tras_vec_2[1]);// + lambda * robot_opt[2];
        residuals[2] = (tras_vec_1[2] - tras_vec_2[2]);
        residuals[3] = (rot_vec_1[0] - rot_vec_2[0]);//  + lambda * robot_opt[0];
        residuals[4] = (rot_vec_1[1] - rot_vec_2[1]);
        residuals[5] = (rot_vec_1[2] - rot_vec_2[2]);
        residuals[6] = tras_vec_3[2] - abs(camz_i);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const Eigen::Vector3d &pnp_r_vec_i_1,
                                        const Eigen::Vector3d &pnp_t_vec_i_1,
                                        const Eigen::Vector3d &robot_r_vec_i_1,
                                        const Eigen::Vector3d &robot_t_vec_i_1,
                                        const Eigen::Vector3d &pnp_r_vec_i,
                                        const Eigen::Vector3d &pnp_t_vec_i,
                                        const Eigen::Vector3d &robot_r_vec_i,
                                        const Eigen::Vector3d &robot_t_vec_i,
                                        const double &camz_i)
    {
        return new ceres::NumericDiffCostFunction<ax_xb_opt, ceres::CENTRAL, 7, 6, 3>(
                new ax_xb_opt(pnp_r_vec_i_1, pnp_t_vec_i_1, robot_r_vec_i_1, robot_t_vec_i_1, pnp_r_vec_i, pnp_t_vec_i, robot_r_vec_i, robot_t_vec_i, camz_i));
    }

    Eigen::Vector3d pnp_r_vec_i_1;
    Eigen::Vector3d pnp_t_vec_i_1;
    Eigen::Vector3d robot_r_vec_i_1;
    Eigen::Vector3d robot_t_vec_i_1;
    Eigen::Vector3d pnp_r_vec_i;
    Eigen::Vector3d pnp_t_vec_i;
    Eigen::Vector3d robot_r_vec_i;
    Eigen::Vector3d robot_t_vec_i;
    double camz_i;
};




struct Classic_ax_xb_wo_z
{
    Classic_ax_xb_wo_z(const Eigen::Vector3d &pnp_r_vec_i_1,
                  const Eigen::Vector3d &pnp_t_vec_i_1,
                  const Eigen::Vector3d &robot_r_vec_i_1,
                  const Eigen::Vector3d &robot_t_vec_i_1,
                  const Eigen::Vector3d &pnp_r_vec_i,
                  const Eigen::Vector3d &pnp_t_vec_i,
                  const Eigen::Vector3d &robot_r_vec_i,
                  const Eigen::Vector3d &robot_t_vec_i) :
            pnp_r_vec_i_1(pnp_r_vec_i_1),
            pnp_t_vec_i_1(pnp_t_vec_i_1),
            robot_r_vec_i_1(robot_r_vec_i_1),
            robot_t_vec_i_1(robot_t_vec_i_1),
            pnp_r_vec_i(pnp_r_vec_i),
            pnp_t_vec_i(pnp_t_vec_i),
            robot_r_vec_i(robot_r_vec_i),
            robot_t_vec_i(robot_t_vec_i){}

    bool operator()(const double* const h2e,
                    double* residuals) const
    {
        double tcp_pt[3], tcp_pt2[3], base_r[3], cam_pt_rob[3], cam_pt_pnp[3], cam_pt2_pnp[3], ee_pt2_rob[3], rot_mat[9], rot_mat_inv_1[9], rot_vec_1[3], tras_vec_1[3], rot_mat2[9], rot_mat_inv2[9], rot_vec_2[3], tras_vec_2[3],
                cal_rot_mat_i[9], cal_rot_mat_i_1[9], h2e_rot_mat[9], rot_mat_inv_2[9], rot_mat_rob_i_1[9], rot_mat_rob_i[9], opt_rot_mat_i_1[9], opt_rot_mat_i[9], tras_vec_3[3], tras_vec_4[3];

        ceres::AngleAxisToRotationMatrix(h2e, h2e_rot_mat);
        cv::Mat X = (cv::Mat_<double>(4,4) <<
                h2e_rot_mat[0], h2e_rot_mat[3], h2e_rot_mat[6], h2e[3],
                h2e_rot_mat[1], h2e_rot_mat[4], h2e_rot_mat[7], h2e[4],
                h2e_rot_mat[2], h2e_rot_mat[5], h2e_rot_mat[8], h2e[5],
                0, 0, 0, 1);


        double robot_r_i_1[3] = {double(robot_r_vec_i_1(0)), double(robot_r_vec_i_1(1)), double(robot_r_vec_i_1(2))},
                robot_t_i_1[3] = {double(robot_t_vec_i_1(0)), double(robot_t_vec_i_1(1)), double(robot_t_vec_i_1(2))};

        double robot_r_i[3] = {double(robot_r_vec_i(0)), double(robot_r_vec_i(1)), double(robot_r_vec_i(2))},
                robot_t_i[3] = {double(robot_t_vec_i(0)), double(robot_t_vec_i(1)), double(robot_t_vec_i(2))};


        //###########################################################################################
        ceres::AngleAxisToRotationMatrix(robot_r_i_1, rot_mat_rob_i_1);

        cv::Mat A_i_1 = (cv::Mat_<double>(4,4) <<
                                               rot_mat_rob_i_1[0], rot_mat_rob_i_1[3], rot_mat_rob_i_1[6], robot_t_i_1[0],
                rot_mat_rob_i_1[1], rot_mat_rob_i_1[4], rot_mat_rob_i_1[7], robot_t_i_1[1],
                rot_mat_rob_i_1[2], rot_mat_rob_i_1[5], rot_mat_rob_i_1[8], robot_t_i_1[2],
                0, 0, 0, 1);

        cv::Mat A_i_1_inv = A_i_1.inv();

        ceres::AngleAxisToRotationMatrix(robot_r_i, rot_mat_rob_i);

        cv::Mat A_i = (cv::Mat_<double>(4,4) <<
                                             rot_mat_rob_i[0], rot_mat_rob_i[3], rot_mat_rob_i[6], robot_t_i[0],
                rot_mat_rob_i[1], rot_mat_rob_i[4], rot_mat_rob_i[7], robot_t_i[1],
                rot_mat_rob_i[2], rot_mat_rob_i[5], rot_mat_rob_i[8], robot_t_i[2],
                0, 0, 0, 1);


        // T^{C}_{EE} * T^{EE}_{E} * T^{E}_{W}i-1*{W}_{E}i * T^{E}_{EE}
        cv::Mat chain1 = A_i_1_inv * A_i * X;


        rot_mat_inv_1[0] = chain1.at<double>(0,0);
        rot_mat_inv_1[1] = chain1.at<double>(1,0);
        rot_mat_inv_1[2] = chain1.at<double>(2,0);
        rot_mat_inv_1[3] = chain1.at<double>(0,1);
        rot_mat_inv_1[4] = chain1.at<double>(1,1);
        rot_mat_inv_1[5] = chain1.at<double>(2,1);
        rot_mat_inv_1[6] = chain1.at<double>(0,2);
        rot_mat_inv_1[7] = chain1.at<double>(1,2);
        rot_mat_inv_1[8] = chain1.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_1, rot_vec_1);

        tras_vec_1[0] = chain1.at<double>(0,3);
        tras_vec_1[1] = chain1.at<double>(1,3);
        tras_vec_1[2] = chain1.at<double>(2,3);


        //###########################################################################################

        // Chain 2
        double pnp_r_i_1[3] = {double(pnp_r_vec_i_1(0)), double(pnp_r_vec_i_1(1)), double(pnp_r_vec_i_1(2))};
        double pnp_t_i_1[3] = {double(pnp_t_vec_i_1(0)), double(pnp_t_vec_i_1(1)), double(pnp_t_vec_i_1(2))};

        double pnp_r_i[3] = {double(pnp_r_vec_i(0)), double(pnp_r_vec_i(1)), double(pnp_r_vec_i(2))};
        double pnp_t_i[3] = {double(pnp_t_vec_i(0)), double(pnp_t_vec_i(1)), double(pnp_t_vec_i(2))};

        ceres::AngleAxisToRotationMatrix(pnp_r_i, cal_rot_mat_i);
        cv::Mat B_i = (cv::Mat_<double>(4,4) <<
                                             cal_rot_mat_i[0], cal_rot_mat_i[3], cal_rot_mat_i[6], pnp_t_i[0],
                cal_rot_mat_i[1], cal_rot_mat_i[4], cal_rot_mat_i[7], pnp_t_i[1],
                cal_rot_mat_i[2], cal_rot_mat_i[5], cal_rot_mat_i[8], pnp_t_i[2],
                0, 0, 0, 1);

        cv::Mat B_i_inv = B_i.inv();

        ceres::AngleAxisToRotationMatrix(pnp_r_i_1, cal_rot_mat_i_1);
        cv::Mat B_i_1 = (cv::Mat_<double>(4,4) <<
                                               cal_rot_mat_i_1[0], cal_rot_mat_i_1[3], cal_rot_mat_i_1[6], pnp_t_i_1[0],
                cal_rot_mat_i_1[1], cal_rot_mat_i_1[4], cal_rot_mat_i_1[7], pnp_t_i_1[1],
                cal_rot_mat_i_1[2], cal_rot_mat_i_1[5], cal_rot_mat_i_1[8], pnp_t_i_1[2],
                0, 0, 0, 1);



        // T^{C}_{B}i-1 * T^{B}_{C}i * T^{C}_{EE}
        cv::Mat chain2 = X * B_i_1 * B_i_inv;

        rot_mat_inv_2[0] = chain2.at<double>(0,0);
        rot_mat_inv_2[1] = chain2.at<double>(1,0);
        rot_mat_inv_2[2] = chain2.at<double>(2,0);
        rot_mat_inv_2[3] = chain2.at<double>(0,1);
        rot_mat_inv_2[4] = chain2.at<double>(1,1);
        rot_mat_inv_2[5] = chain2.at<double>(2,1);
        rot_mat_inv_2[6] = chain2.at<double>(0,2);
        rot_mat_inv_2[7] = chain2.at<double>(1,2);
        rot_mat_inv_2[8] = chain2.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_2, rot_vec_2);

        tras_vec_2[0] = chain2.at<double>(0,3);
        tras_vec_2[1] = chain2.at<double>(1,3);
        tras_vec_2[2] = chain2.at<double>(2,3);

        //###########################################################################################

        residuals[0] = (tras_vec_1[0] - tras_vec_2[0]);
        residuals[1] = (tras_vec_1[1] - tras_vec_2[1]);
        residuals[2] = (tras_vec_1[2] - tras_vec_2[2]);
        residuals[3] = (rot_vec_1[0] - rot_vec_2[0]);
        residuals[4] = (rot_vec_1[1] - rot_vec_2[1]);
        residuals[5] = (rot_vec_1[2] - rot_vec_2[2]);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const Eigen::Vector3d &pnp_r_vec_i_1,
                                        const Eigen::Vector3d &pnp_t_vec_i_1,
                                        const Eigen::Vector3d &robot_r_vec_i_1,
                                        const Eigen::Vector3d &robot_t_vec_i_1,
                                        const Eigen::Vector3d &pnp_r_vec_i,
                                        const Eigen::Vector3d &pnp_t_vec_i,
                                        const Eigen::Vector3d &robot_r_vec_i,
                                        const Eigen::Vector3d &robot_t_vec_i)
    {
        return new ceres::NumericDiffCostFunction<Classic_ax_xb_wo_z, ceres::CENTRAL, 6, 6>(
                new Classic_ax_xb_wo_z(pnp_r_vec_i_1, pnp_t_vec_i_1, robot_r_vec_i_1, robot_t_vec_i_1, pnp_r_vec_i, pnp_t_vec_i, robot_r_vec_i, robot_t_vec_i));
    }

    Eigen::Vector3d pnp_r_vec_i_1;
    Eigen::Vector3d pnp_t_vec_i_1;
    Eigen::Vector3d robot_r_vec_i_1;
    Eigen::Vector3d robot_t_vec_i_1;
    Eigen::Vector3d pnp_r_vec_i;
    Eigen::Vector3d pnp_t_vec_i;
    Eigen::Vector3d robot_r_vec_i;
    Eigen::Vector3d robot_t_vec_i;
};




struct Classic_ax_xb_wo_z_rel
{
    Classic_ax_xb_wo_z_rel(const Eigen::Vector3d &pnp_r_rel,
                       const Eigen::Vector3d &pnp_t_rel,
                       const Eigen::Vector3d &robot_r_rel,
                       const Eigen::Vector3d &robot_t_rel) :
            pnp_r_rel(pnp_r_rel),
            pnp_t_rel(pnp_t_rel),
            robot_r_rel(robot_r_rel),
            robot_t_rel(robot_t_rel){}

    bool operator()(const double* const h2e,
                    double* residuals) const
    {
        double tcp_pt[3], tcp_pt2[3], base_r[3], cam_pt_rob[3], cam_pt_pnp[3], cam_pt2_pnp[3], ee_pt2_rob[3], rot_mat[9], rot_mat_inv_1[9], rot_vec_1[3], tras_vec_1[3], rot_mat2[9], rot_mat_inv2[9], rot_vec_2[3], tras_vec_2[3],
                cal_rot_mat[9], h2e_rot_mat[9], rot_mat_inv_2[9], rot_mat_rob[9], opt_rot_mat_i_1[9], opt_rot_mat_i[9], tras_vec_3[3], tras_vec_4[3];

        ceres::AngleAxisToRotationMatrix(h2e, h2e_rot_mat);
        cv::Mat X = (cv::Mat_<double>(4,4) <<
                h2e_rot_mat[0], h2e_rot_mat[3], h2e_rot_mat[6], h2e[3],
                h2e_rot_mat[1], h2e_rot_mat[4], h2e_rot_mat[7], h2e[4],
                h2e_rot_mat[2], h2e_rot_mat[5], h2e_rot_mat[8], h2e[5],
                0, 0, 0, 1);


        double robot_r[3] = {double(robot_r_rel(0)), double(robot_r_rel(1)), double(robot_r_rel(2))},
                robot_t[3] = {double(robot_t_rel(0)), double(robot_t_rel(1)), double(robot_t_rel(2))};

        //###########################################################################################
        ceres::AngleAxisToRotationMatrix(robot_r, rot_mat_rob);

        cv::Mat A = (cv::Mat_<double>(4,4) <<
                rot_mat_rob[0], rot_mat_rob[3], rot_mat_rob[6], robot_t[0],
                rot_mat_rob[1], rot_mat_rob[4], rot_mat_rob[7], robot_t[1],
                rot_mat_rob[2], rot_mat_rob[5], rot_mat_rob[8], robot_t[2],
                0, 0, 0, 1);


        cv::Mat chain1 = A * X;

        rot_mat_inv_1[0] = chain1.at<double>(0,0);
        rot_mat_inv_1[1] = chain1.at<double>(1,0);
        rot_mat_inv_1[2] = chain1.at<double>(2,0);
        rot_mat_inv_1[3] = chain1.at<double>(0,1);
        rot_mat_inv_1[4] = chain1.at<double>(1,1);
        rot_mat_inv_1[5] = chain1.at<double>(2,1);
        rot_mat_inv_1[6] = chain1.at<double>(0,2);
        rot_mat_inv_1[7] = chain1.at<double>(1,2);
        rot_mat_inv_1[8] = chain1.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_1, rot_vec_1);

        tras_vec_1[0] = chain1.at<double>(0,3);
        tras_vec_1[1] = chain1.at<double>(1,3);
        tras_vec_1[2] = chain1.at<double>(2,3);


        //###########################################################################################

        // Chain 2
        double pnp_r[3] = {double(pnp_r_rel(0)), double(pnp_r_rel(1)), double(pnp_r_rel(2))};
        double pnp_t[3] = {double(pnp_t_rel(0)), double(pnp_t_rel(1)), double(pnp_t_rel(2))};

        ceres::AngleAxisToRotationMatrix(pnp_r, cal_rot_mat);
        cv::Mat B = (cv::Mat_<double>(4,4) <<
                cal_rot_mat[0], cal_rot_mat[3], cal_rot_mat[6], pnp_t[0],
                cal_rot_mat[1], cal_rot_mat[4], cal_rot_mat[7], pnp_t[1],
                cal_rot_mat[2], cal_rot_mat[5], cal_rot_mat[8], pnp_t[2],
                0, 0, 0, 1);


        // T^{C}_{B}i-1 * T^{B}_{C}i * T^{C}_{EE}
        cv::Mat chain2 = X * B;

        rot_mat_inv_2[0] = chain2.at<double>(0,0);
        rot_mat_inv_2[1] = chain2.at<double>(1,0);
        rot_mat_inv_2[2] = chain2.at<double>(2,0);
        rot_mat_inv_2[3] = chain2.at<double>(0,1);
        rot_mat_inv_2[4] = chain2.at<double>(1,1);
        rot_mat_inv_2[5] = chain2.at<double>(2,1);
        rot_mat_inv_2[6] = chain2.at<double>(0,2);
        rot_mat_inv_2[7] = chain2.at<double>(1,2);
        rot_mat_inv_2[8] = chain2.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_2, rot_vec_2);

        tras_vec_2[0] = chain2.at<double>(0,3);
        tras_vec_2[1] = chain2.at<double>(1,3);
        tras_vec_2[2] = chain2.at<double>(2,3);

        //###########################################################################################

        residuals[0] = (tras_vec_1[0] - tras_vec_2[0]);
        residuals[1] = (tras_vec_1[1] - tras_vec_2[1]);
        residuals[2] = (tras_vec_1[2] - tras_vec_2[2]);
        residuals[3] = (rot_vec_1[0] - rot_vec_2[0]);
        residuals[4] = (rot_vec_1[1] - rot_vec_2[1]);
        residuals[5] = (rot_vec_1[2] - rot_vec_2[2]);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const Eigen::Vector3d &pnp_r_rel,
                                        const Eigen::Vector3d &pnp_t_rel,
                                        const Eigen::Vector3d &robot_r_rel,
                                        const Eigen::Vector3d &robot_t_rel)
    {
        return new ceres::NumericDiffCostFunction<Classic_ax_xb_wo_z_rel, ceres::CENTRAL, 6, 6>(
                new Classic_ax_xb_wo_z_rel(pnp_r_rel, pnp_t_rel, robot_r_rel, robot_t_rel));
    }

    Eigen::Vector3d pnp_r_rel;
    Eigen::Vector3d pnp_t_rel;
    Eigen::Vector3d robot_r_rel;
    Eigen::Vector3d robot_t_rel;
};



struct Classic_ax_xb_wo_z_multi
{
    Classic_ax_xb_wo_z_multi(const Eigen::Vector3d &pnp_r_vec_i_1_c1,
                       const Eigen::Vector3d &pnp_t_vec_i_1_c1,
                       const Eigen::Vector3d &pnp_r_vec_i_c1,
                       const Eigen::Vector3d &pnp_t_vec_i_c1,
                       const Eigen::Vector3d &pnp_r_vec_i_1_c2,
                       const Eigen::Vector3d &pnp_t_vec_i_1_c2,
                       const Eigen::Vector3d &pnp_r_vec_i_c2,
                       const Eigen::Vector3d &pnp_t_vec_i_c2,
                       const Eigen::Vector3d &robot_r_vec_i_1,
                       const Eigen::Vector3d &robot_t_vec_i_1,
                       const Eigen::Vector3d &robot_r_vec_i,
                       const Eigen::Vector3d &robot_t_vec_i) :
            pnp_r_vec_i_1_c1(pnp_r_vec_i_1_c1),
            pnp_t_vec_i_1_c1(pnp_t_vec_i_1_c1),
            pnp_r_vec_i_c1(pnp_r_vec_i_c1),
            pnp_t_vec_i_c1(pnp_t_vec_i_c1),
            pnp_r_vec_i_1_c2(pnp_r_vec_i_1_c2),
            pnp_t_vec_i_1_c2(pnp_t_vec_i_1_c2),
            pnp_r_vec_i_c2(pnp_r_vec_i_c2),
            pnp_t_vec_i_c2(pnp_t_vec_i_c2),
            robot_r_vec_i_1(robot_r_vec_i_1),
            robot_t_vec_i_1(robot_t_vec_i_1),
            robot_r_vec_i(robot_r_vec_i),
            robot_t_vec_i(robot_t_vec_i){}

    bool operator()(const double* const h2e1,
                    const double* const h2e2,
                    const double* const c2c,
                    double* residuals) const
    {
        double  rot_mat_inv_1[9], rot_mat_inv_2[9], rot_mat_inv_ch2_1[9], rot_mat_inv_ch2_2[9], rot_vec_ch1_1[3], rot_vec_ch1_2[3], rot_vec_ch2_1[3], rot_vec_ch2_2[3], tras_vec_ch1_1[3], tras_vec_ch1_2[3], tras_vec_ch2_1[3], tras_vec_ch2_2[3],
                cal_rot_mat_i_c1[9], cal_rot_mat_i_1_c1[9], cal_rot_mat_i_c2[9], cal_rot_mat_i_1_c2[9], c2c_rot_mat[9], h2e1_rot_mat[9], h2e2_rot_mat[9], rot_mat_rob_i_1[9], rot_mat_rob_i[9], opt_rot_mat_i_1[9], opt_rot_mat_i[9], tras_vec_3[3], tras_vec_4[3];

        ceres::AngleAxisToRotationMatrix(h2e1, h2e1_rot_mat);
        cv::Mat X1 = (cv::Mat_<double>(4,4) <<
                h2e1_rot_mat[0], h2e1_rot_mat[3], h2e1_rot_mat[6], h2e1[3],
                h2e1_rot_mat[1], h2e1_rot_mat[4], h2e1_rot_mat[7], h2e1[4],
                h2e1_rot_mat[2], h2e1_rot_mat[5], h2e1_rot_mat[8], h2e1[5],
                0, 0, 0, 1);

        ceres::AngleAxisToRotationMatrix(h2e2, h2e2_rot_mat);
        cv::Mat X2 = (cv::Mat_<double>(4,4) <<
                h2e2_rot_mat[0], h2e2_rot_mat[3], h2e2_rot_mat[6], h2e2[3],
                h2e2_rot_mat[1], h2e2_rot_mat[4], h2e2_rot_mat[7], h2e2[4],
                h2e2_rot_mat[2], h2e2_rot_mat[5], h2e2_rot_mat[8], h2e2[5],
                0, 0, 0, 1);

        ceres::AngleAxisToRotationMatrix(c2c, c2c_rot_mat);
        cv::Mat Y = (cv::Mat_<double>(4,4) <<
                c2c_rot_mat[0], c2c_rot_mat[3], c2c_rot_mat[6], c2c[3],
                c2c_rot_mat[1], c2c_rot_mat[4], c2c_rot_mat[7], c2c[4],
                c2c_rot_mat[2], c2c_rot_mat[5], c2c_rot_mat[8], c2c[5],
                0, 0, 0, 1);


        double robot_r_i_1[3] = {double(robot_r_vec_i_1(0)), double(robot_r_vec_i_1(1)), double(robot_r_vec_i_1(2))},
                robot_t_i_1[3] = {double(robot_t_vec_i_1(0)), double(robot_t_vec_i_1(1)), double(robot_t_vec_i_1(2))};

        double robot_r_i[3] = {double(robot_r_vec_i(0)), double(robot_r_vec_i(1)), double(robot_r_vec_i(2))},
                robot_t_i[3] = {double(robot_t_vec_i(0)), double(robot_t_vec_i(1)), double(robot_t_vec_i(2))};


        //###########################################################################################
        ceres::AngleAxisToRotationMatrix(robot_r_i_1, rot_mat_rob_i_1);

        cv::Mat A_i_1 = (cv::Mat_<double>(4,4) <<
                rot_mat_rob_i_1[0], rot_mat_rob_i_1[3], rot_mat_rob_i_1[6], robot_t_i_1[0],
                rot_mat_rob_i_1[1], rot_mat_rob_i_1[4], rot_mat_rob_i_1[7], robot_t_i_1[1],
                rot_mat_rob_i_1[2], rot_mat_rob_i_1[5], rot_mat_rob_i_1[8], robot_t_i_1[2],
                0, 0, 0, 1);

        cv::Mat A_i_1_inv = A_i_1.inv();

        ceres::AngleAxisToRotationMatrix(robot_r_i, rot_mat_rob_i);

        cv::Mat A_i = (cv::Mat_<double>(4,4) <<
                rot_mat_rob_i[0], rot_mat_rob_i[3], rot_mat_rob_i[6], robot_t_i[0],
                rot_mat_rob_i[1], rot_mat_rob_i[4], rot_mat_rob_i[7], robot_t_i[1],
                rot_mat_rob_i[2], rot_mat_rob_i[5], rot_mat_rob_i[8], robot_t_i[2],
                0, 0, 0, 1);

        cv::Mat A = A_i_1_inv * A_i;

        cv::Mat chain1_c1 = A * X1 * Y;
        cv::Mat chain1_c2 = A * X2 * Y.inv();


        rot_mat_inv_1[0] = chain1_c1.at<double>(0,0);
        rot_mat_inv_1[1] = chain1_c1.at<double>(1,0);
        rot_mat_inv_1[2] = chain1_c1.at<double>(2,0);
        rot_mat_inv_1[3] = chain1_c1.at<double>(0,1);
        rot_mat_inv_1[4] = chain1_c1.at<double>(1,1);
        rot_mat_inv_1[5] = chain1_c1.at<double>(2,1);
        rot_mat_inv_1[6] = chain1_c1.at<double>(0,2);
        rot_mat_inv_1[7] = chain1_c1.at<double>(1,2);
        rot_mat_inv_1[8] = chain1_c1.at<double>(2,2);

        rot_mat_inv_2[0] = chain1_c2.at<double>(0,0);
        rot_mat_inv_2[1] = chain1_c2.at<double>(1,0);
        rot_mat_inv_2[2] = chain1_c2.at<double>(2,0);
        rot_mat_inv_2[3] = chain1_c2.at<double>(0,1);
        rot_mat_inv_2[4] = chain1_c2.at<double>(1,1);
        rot_mat_inv_2[5] = chain1_c2.at<double>(2,1);
        rot_mat_inv_2[6] = chain1_c2.at<double>(0,2);
        rot_mat_inv_2[7] = chain1_c2.at<double>(1,2);
        rot_mat_inv_2[8] = chain1_c2.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_1, rot_vec_ch1_1);
        ceres::RotationMatrixToAngleAxis(rot_mat_inv_2, rot_vec_ch1_2);

        tras_vec_ch1_1[0] = chain1_c1.at<double>(0,3);
        tras_vec_ch1_1[1] = chain1_c1.at<double>(1,3);
        tras_vec_ch1_1[2] = chain1_c1.at<double>(2,3);

        tras_vec_ch1_2[0] = chain1_c2.at<double>(0,3);
        tras_vec_ch1_2[1] = chain1_c2.at<double>(1,3);
        tras_vec_ch1_2[2] = chain1_c2.at<double>(2,3);


        //###########################################################################################

        // Chain 2
        double pnp_r_i_1_c1[3] = {double(pnp_r_vec_i_1_c1(0)), double(pnp_r_vec_i_1_c1(1)), double(pnp_r_vec_i_1_c1(2))};
        double pnp_t_i_1_c1[3] = {double(pnp_t_vec_i_1_c1(0)), double(pnp_t_vec_i_1_c1(1)), double(pnp_t_vec_i_1_c1(2))};

        double pnp_r_i_c1[3] = {double(pnp_r_vec_i_c1(0)), double(pnp_r_vec_i_c1(1)), double(pnp_r_vec_i_c1(2))};
        double pnp_t_i_c1[3] = {double(pnp_t_vec_i_c1(0)), double(pnp_t_vec_i_c1(1)), double(pnp_t_vec_i_c1(2))};

        double pnp_r_i_1_c2[3] = {double(pnp_r_vec_i_1_c2(0)), double(pnp_r_vec_i_1_c2(1)), double(pnp_r_vec_i_1_c2(2))};
        double pnp_t_i_1_c2[3] = {double(pnp_t_vec_i_1_c2(0)), double(pnp_t_vec_i_1_c2(1)), double(pnp_t_vec_i_1_c2(2))};

        double pnp_r_i_c2[3] = {double(pnp_r_vec_i_c2(0)), double(pnp_r_vec_i_c2(1)), double(pnp_r_vec_i_c2(2))};
        double pnp_t_i_c2[3] = {double(pnp_t_vec_i_c2(0)), double(pnp_t_vec_i_c2(1)), double(pnp_t_vec_i_c2(2))};


        ceres::AngleAxisToRotationMatrix(pnp_r_i_c1, cal_rot_mat_i_c1);
        cv::Mat B_i_c1 = (cv::Mat_<double>(4,4) <<
                                             cal_rot_mat_i_c1[0], cal_rot_mat_i_c1[3], cal_rot_mat_i_c1[6], pnp_t_i_c1[0],
                cal_rot_mat_i_c1[1], cal_rot_mat_i_c1[4], cal_rot_mat_i_c1[7], pnp_t_i_c1[1],
                cal_rot_mat_i_c1[2], cal_rot_mat_i_c1[5], cal_rot_mat_i_c1[8], pnp_t_i_c1[2],
                0, 0, 0, 1);

        cv::Mat B_i_inv_c1 = B_i_c1.inv();

        ceres::AngleAxisToRotationMatrix(pnp_r_i_1_c1, cal_rot_mat_i_1_c1);
        cv::Mat B_i_1_c1 = (cv::Mat_<double>(4,4) <<
                                               cal_rot_mat_i_1_c1[0], cal_rot_mat_i_1_c1[3], cal_rot_mat_i_1_c1[6], pnp_t_i_1_c1[0],
                cal_rot_mat_i_1_c1[1], cal_rot_mat_i_1_c1[4], cal_rot_mat_i_1_c1[7], pnp_t_i_1_c1[1],
                cal_rot_mat_i_1_c1[2], cal_rot_mat_i_1_c1[5], cal_rot_mat_i_1_c1[8], pnp_t_i_1_c1[2],
                0, 0, 0, 1);



        ceres::AngleAxisToRotationMatrix(pnp_r_i_c2, cal_rot_mat_i_c2);
        cv::Mat B_i_c2 = (cv::Mat_<double>(4,4) <<
                cal_rot_mat_i_c2[0], cal_rot_mat_i_c2[3], cal_rot_mat_i_c2[6], pnp_t_i_c2[0],
                cal_rot_mat_i_c2[1], cal_rot_mat_i_c2[4], cal_rot_mat_i_c2[7], pnp_t_i_c2[1],
                cal_rot_mat_i_c2[2], cal_rot_mat_i_c2[5], cal_rot_mat_i_c2[8], pnp_t_i_c2[2],
                0, 0, 0, 1);

        cv::Mat B_i_inv_c2 = B_i_c2.inv();

        ceres::AngleAxisToRotationMatrix(pnp_r_i_1_c2, cal_rot_mat_i_1_c2);
        cv::Mat B_i_1_c2 = (cv::Mat_<double>(4,4) <<
                cal_rot_mat_i_1_c2[0], cal_rot_mat_i_1_c2[3], cal_rot_mat_i_1_c2[6], pnp_t_i_1_c2[0],
                cal_rot_mat_i_1_c2[1], cal_rot_mat_i_1_c2[4], cal_rot_mat_i_1_c2[7], pnp_t_i_1_c2[1],
                cal_rot_mat_i_1_c2[2], cal_rot_mat_i_1_c2[5], cal_rot_mat_i_1_c2[8], pnp_t_i_1_c2[2],
                0, 0, 0, 1);

        cv::Mat B1 = B_i_1_c1 * B_i_inv_c1;
        cv::Mat B2 = B_i_1_c2 * B_i_inv_c2;
        cv::Mat chain2_c1 = X1 * B1;
        cv::Mat chain2_c2 = X2 * B2;

        rot_mat_inv_ch2_1[0] = chain2_c1.at<double>(0,0);
        rot_mat_inv_ch2_1[1] = chain2_c1.at<double>(1,0);
        rot_mat_inv_ch2_1[2] = chain2_c1.at<double>(2,0);
        rot_mat_inv_ch2_1[3] = chain2_c1.at<double>(0,1);
        rot_mat_inv_ch2_1[4] = chain2_c1.at<double>(1,1);
        rot_mat_inv_ch2_1[5] = chain2_c1.at<double>(2,1);
        rot_mat_inv_ch2_1[6] = chain2_c1.at<double>(0,2);
        rot_mat_inv_ch2_1[7] = chain2_c1.at<double>(1,2);
        rot_mat_inv_ch2_1[8] = chain2_c1.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_ch2_1, rot_vec_ch2_1);

        tras_vec_ch2_1[0] = chain2_c1.at<double>(0,3);
        tras_vec_ch2_1[1] = chain2_c1.at<double>(1,3);
        tras_vec_ch2_1[2] = chain2_c1.at<double>(2,3);


        rot_mat_inv_ch2_2[0] = chain2_c2.at<double>(0,0);
        rot_mat_inv_ch2_2[1] = chain2_c2.at<double>(1,0);
        rot_mat_inv_ch2_2[2] = chain2_c2.at<double>(2,0);
        rot_mat_inv_ch2_2[3] = chain2_c2.at<double>(0,1);
        rot_mat_inv_ch2_2[4] = chain2_c2.at<double>(1,1);
        rot_mat_inv_ch2_2[5] = chain2_c2.at<double>(2,1);
        rot_mat_inv_ch2_2[6] = chain2_c2.at<double>(0,2);
        rot_mat_inv_ch2_2[7] = chain2_c2.at<double>(1,2);
        rot_mat_inv_ch2_2[8] = chain2_c2.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_ch2_2, rot_vec_ch2_2);

        tras_vec_ch2_2[0] = chain2_c2.at<double>(0,3);
        tras_vec_ch2_2[1] = chain2_c2.at<double>(1,3);
        tras_vec_ch2_2[2] = chain2_c2.at<double>(2,3);

        //###########################################################################################

        residuals[0] = (tras_vec_ch1_1[0] - tras_vec_ch2_2[0]);
        residuals[1] = (tras_vec_ch1_1[1] - tras_vec_ch2_2[1]);
        residuals[2] = (tras_vec_ch1_1[2] - tras_vec_ch2_2[2]);

        residuals[3] = (tras_vec_ch1_2[0] - tras_vec_ch2_1[0]);
        residuals[4] = (tras_vec_ch1_2[1] - tras_vec_ch2_1[1]);
        residuals[5] = (tras_vec_ch1_2[2] - tras_vec_ch2_1[2]);

        residuals[6] = (rot_vec_ch1_1[0] - rot_vec_ch2_2[0]);
        residuals[7] = (rot_vec_ch1_1[1] - rot_vec_ch2_2[1]);
        residuals[8] = (rot_vec_ch1_1[2] - rot_vec_ch2_2[2]);

        residuals[9] = (rot_vec_ch1_2[0] - rot_vec_ch2_1[0]);
        residuals[10] = (rot_vec_ch1_2[1] - rot_vec_ch2_1[1]);
        residuals[11] = (rot_vec_ch1_2[2] - rot_vec_ch2_1[2]);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const Eigen::Vector3d &pnp_r_vec_i_1_c1,
                                       const Eigen::Vector3d &pnp_t_vec_i_1_c1,
                                       const Eigen::Vector3d &pnp_r_vec_i_c1,
                                       const Eigen::Vector3d &pnp_t_vec_i_c1,
                                       const Eigen::Vector3d &pnp_r_vec_i_1_c2,
                                       const Eigen::Vector3d &pnp_t_vec_i_1_c2,
                                       const Eigen::Vector3d &pnp_r_vec_i_c2,
                                       const Eigen::Vector3d &pnp_t_vec_i_c2,
                                       const Eigen::Vector3d &robot_r_vec_i_1,
                                       const Eigen::Vector3d &robot_t_vec_i_1,
                                       const Eigen::Vector3d &robot_r_vec_i,
                                       const Eigen::Vector3d &robot_t_vec_i)
    {
        return new ceres::NumericDiffCostFunction<Classic_ax_xb_wo_z_multi, ceres::CENTRAL, 12, 6, 6, 6>(
                new Classic_ax_xb_wo_z_multi(pnp_r_vec_i_1_c1, pnp_t_vec_i_1_c1, pnp_r_vec_i_c1, pnp_t_vec_i_c1, pnp_r_vec_i_1_c2,
                                             pnp_t_vec_i_1_c2, pnp_r_vec_i_c2, pnp_t_vec_i_c2, robot_r_vec_i_1, robot_t_vec_i_1,
                                             robot_r_vec_i, robot_t_vec_i));
    }

    Eigen::Vector3d pnp_r_vec_i_1_c1;
    Eigen::Vector3d pnp_t_vec_i_1_c1;
    Eigen::Vector3d pnp_r_vec_i_c1;
    Eigen::Vector3d pnp_t_vec_i_c1;
    Eigen::Vector3d pnp_r_vec_i_1_c2;
    Eigen::Vector3d pnp_t_vec_i_1_c2;
    Eigen::Vector3d pnp_r_vec_i_c2;
    Eigen::Vector3d pnp_t_vec_i_c2;
    Eigen::Vector3d robot_r_vec_i_1;
    Eigen::Vector3d robot_t_vec_i_1;
    Eigen::Vector3d robot_r_vec_i;
    Eigen::Vector3d robot_t_vec_i;
};




struct AX_XB
{
    AX_XB(const Eigen::Vector3d &pnp_r_vec_i_1,
                  const Eigen::Vector3d &pnp_t_vec_i_1,
                  const Eigen::Vector3d &robot_r_vec_i_1,
                  const Eigen::Vector3d &robot_t_vec_i_1,
                  const Eigen::Vector3d &pnp_r_vec_i,
                  const Eigen::Vector3d &pnp_t_vec_i,
                  const Eigen::Vector3d &robot_r_vec_i,
                  const Eigen::Vector3d &robot_t_vec_i) :
            pnp_r_vec_i_1(pnp_r_vec_i_1),
            pnp_t_vec_i_1(pnp_t_vec_i_1),
            robot_r_vec_i_1(robot_r_vec_i_1),
            robot_t_vec_i_1(robot_t_vec_i_1),
            pnp_r_vec_i(pnp_r_vec_i),
            pnp_t_vec_i(pnp_t_vec_i),
            robot_r_vec_i(robot_r_vec_i),
            robot_t_vec_i(robot_t_vec_i){}

    bool operator()(const double* const h2e,
                    double* residuals) const
    {
        double tcp_pt[3], tcp_pt2[3], base_r[3], cam_pt_rob[3], cam_pt_pnp[3], cam_pt2_pnp[3], ee_pt2_rob[3], rot_mat[9], rot_mat_inv_1[9], rot_vec_1[3], tras_vec_1[3], rot_mat2[9], rot_mat_inv2[9], rot_vec_2[3], tras_vec_2[3],
                cal_rot_mat_i[9], cal_rot_mat_i_1[9], h2e_rot_mat[9], rot_mat_inv_2[9], rot_mat_rob_i_1[9], rot_mat_rob_i[9], opt_rot_mat_i_1[9], opt_rot_mat_i[9], tras_vec_3[3];

        ceres::AngleAxisToRotationMatrix(h2e, h2e_rot_mat);
        cv::Mat X = (cv::Mat_<double>(4,4) <<
                h2e_rot_mat[0], h2e_rot_mat[3], h2e_rot_mat[6], h2e[3],
                h2e_rot_mat[1], h2e_rot_mat[4], h2e_rot_mat[7], h2e[4],
                h2e_rot_mat[2], h2e_rot_mat[5], h2e_rot_mat[8], h2e[5],
                0, 0, 0, 1);


        double robot_r_i_1[3] = {double(robot_r_vec_i_1(0)), double(robot_r_vec_i_1(1)), double(robot_r_vec_i_1(2))},
                robot_t_i_1[3] = {double(robot_t_vec_i_1(0)), double(robot_t_vec_i_1(1)), double(robot_t_vec_i_1(2))};

        double robot_r_i[3] = {double(robot_r_vec_i(0)), double(robot_r_vec_i(1)), double(robot_r_vec_i(2))},
                robot_t_i[3] = {double(robot_t_vec_i(0)), double(robot_t_vec_i(1)), double(robot_t_vec_i(2))};

        //###########################################################################################
        ceres::AngleAxisToRotationMatrix(robot_r_i_1, rot_mat_rob_i_1);

        cv::Mat A_i_1 = (cv::Mat_<double>(4,4) <<
                rot_mat_rob_i_1[0], rot_mat_rob_i_1[3], rot_mat_rob_i_1[6], robot_t_i_1[0],
                rot_mat_rob_i_1[1], rot_mat_rob_i_1[4], rot_mat_rob_i_1[7], robot_t_i_1[1],
                rot_mat_rob_i_1[2], rot_mat_rob_i_1[5], rot_mat_rob_i_1[8], robot_t_i_1[2],
                0, 0, 0, 1);

        cv::Mat A_i_1_inv = A_i_1.inv();

        ceres::AngleAxisToRotationMatrix(robot_r_i, rot_mat_rob_i);

        cv::Mat A_i = (cv::Mat_<double>(4,4) <<
                                             rot_mat_rob_i[0], rot_mat_rob_i[3], rot_mat_rob_i[6], robot_t_i[0],
                rot_mat_rob_i[1], rot_mat_rob_i[4], rot_mat_rob_i[7], robot_t_i[1],
                rot_mat_rob_i[2], rot_mat_rob_i[5], rot_mat_rob_i[8], robot_t_i[2],
                0, 0, 0, 1);


        // T^{C}_{EE} * T^{EE}_{E} * T^{E}_{W}i-1*{W}_{E}i * T^{E}_{EE}
        cv::Mat chain1 = A_i_1_inv * A_i * X;

        rot_mat_inv_1[0] = chain1.at<double>(0,0);
        rot_mat_inv_1[1] = chain1.at<double>(1,0);
        rot_mat_inv_1[2] = chain1.at<double>(2,0);
        rot_mat_inv_1[3] = chain1.at<double>(0,1);
        rot_mat_inv_1[4] = chain1.at<double>(1,1);
        rot_mat_inv_1[5] = chain1.at<double>(2,1);
        rot_mat_inv_1[6] = chain1.at<double>(0,2);
        rot_mat_inv_1[7] = chain1.at<double>(1,2);
        rot_mat_inv_1[8] = chain1.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_1, rot_vec_1);

        tras_vec_1[0] = chain1.at<double>(0,3);
        tras_vec_1[1] = chain1.at<double>(1,3);
        tras_vec_1[2] = chain1.at<double>(2,3);


        //###########################################################################################

        // Chain 2
        double pnp_r_i_1[3] = {double(pnp_r_vec_i_1(0)), double(pnp_r_vec_i_1(1)), double(pnp_r_vec_i_1(2))};
        double pnp_t_i_1[3] = {double(pnp_t_vec_i_1(0)), double(pnp_t_vec_i_1(1)), double(pnp_t_vec_i_1(2))};

        double pnp_r_i[3] = {double(pnp_r_vec_i(0)), double(pnp_r_vec_i(1)), double(pnp_r_vec_i(2))};
        double pnp_t_i[3] = {double(pnp_t_vec_i(0)), double(pnp_t_vec_i(1)), double(pnp_t_vec_i(2))};

        ceres::AngleAxisToRotationMatrix(pnp_r_i, cal_rot_mat_i);
        cv::Mat pnp_i = (cv::Mat_<double>(4,4) <<
                cal_rot_mat_i[0], cal_rot_mat_i[3], cal_rot_mat_i[6], pnp_t_i[0],
                cal_rot_mat_i[1], cal_rot_mat_i[4], cal_rot_mat_i[7], pnp_t_i[1],
                cal_rot_mat_i[2], cal_rot_mat_i[5], cal_rot_mat_i[8], pnp_t_i[2],
                0, 0, 0, 1);

        cv::Mat B_i = pnp_i.inv();

        ceres::AngleAxisToRotationMatrix(pnp_r_i_1, cal_rot_mat_i_1);
        cv::Mat B_i_1 = (cv::Mat_<double>(4,4) <<
                cal_rot_mat_i_1[0], cal_rot_mat_i_1[3], cal_rot_mat_i_1[6], pnp_t_i_1[0],
                cal_rot_mat_i_1[1], cal_rot_mat_i_1[4], cal_rot_mat_i_1[7], pnp_t_i_1[1],
                cal_rot_mat_i_1[2], cal_rot_mat_i_1[5], cal_rot_mat_i_1[8], pnp_t_i_1[2],
                0, 0, 0, 1);



        // T^{C}_{B}i-1 * T^{B}_{C}i * T^{C}_{EE}
        cv::Mat chain2 = X * B_i_1 * B_i;

        rot_mat_inv_2[0] = chain2.at<double>(0,0);
        rot_mat_inv_2[1] = chain2.at<double>(1,0);
        rot_mat_inv_2[2] = chain2.at<double>(2,0);
        rot_mat_inv_2[3] = chain2.at<double>(0,1);
        rot_mat_inv_2[4] = chain2.at<double>(1,1);
        rot_mat_inv_2[5] = chain2.at<double>(2,1);
        rot_mat_inv_2[6] = chain2.at<double>(0,2);
        rot_mat_inv_2[7] = chain2.at<double>(1,2);
        rot_mat_inv_2[8] = chain2.at<double>(2,2);

        ceres::RotationMatrixToAngleAxis(rot_mat_inv_2, rot_vec_2);

        tras_vec_2[0] = chain2.at<double>(0,3);
        tras_vec_2[1] = chain2.at<double>(1,3);
        tras_vec_2[2] = chain2.at<double>(2,3);
        //###########################################################################################

        residuals[0] = (tras_vec_1[0] - tras_vec_2[0]);
        residuals[1] = (tras_vec_1[1] - tras_vec_2[1]);
        residuals[2] = (tras_vec_1[2] - tras_vec_2[2]);
        residuals[3] = (rot_vec_1[0] - rot_vec_2[0]);
        residuals[4] = (rot_vec_1[1] - rot_vec_2[1]);
        residuals[5] = (rot_vec_1[2] - rot_vec_2[2]);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create( const Eigen::Vector3d &pnp_r_vec_i_1,
                                        const Eigen::Vector3d &pnp_t_vec_i_1,
                                        const Eigen::Vector3d &robot_r_vec_i_1,
                                        const Eigen::Vector3d &robot_t_vec_i_1,
                                        const Eigen::Vector3d &pnp_r_vec_i,
                                        const Eigen::Vector3d &pnp_t_vec_i,
                                        const Eigen::Vector3d &robot_r_vec_i,
                                        const Eigen::Vector3d &robot_t_vec_i)
    {
        return new ceres::NumericDiffCostFunction<AX_XB, ceres::CENTRAL, 6, 6>(
                new AX_XB(pnp_r_vec_i_1, pnp_t_vec_i_1, robot_r_vec_i_1, robot_t_vec_i_1, pnp_r_vec_i, pnp_t_vec_i, robot_r_vec_i, robot_t_vec_i));
    }

    Eigen::Vector3d pnp_r_vec_i_1;
    Eigen::Vector3d pnp_t_vec_i_1;
    Eigen::Vector3d robot_r_vec_i_1;
    Eigen::Vector3d robot_t_vec_i_1;
    Eigen::Vector3d pnp_r_vec_i;
    Eigen::Vector3d pnp_t_vec_i;
    Eigen::Vector3d robot_r_vec_i;
    Eigen::Vector3d robot_t_vec_i;
};




#endif //METRIC_CALIBRATOR_HANDEYE_CALIBRATOR_H
