//
// Created by davide on 11/12/23.
//

#ifndef METRIC_CALIBRATOR_MOBILEHANDEYECALIBRATOR_H
#define METRIC_CALIBRATOR_MOBILEHANDEYECALIBRATOR_H

#include <iostream>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>

#include "utils.h"
#include "display_utils.h"
#include "handeye_calibrator.h"
#include "PinholeCameraModel.h"

class MobileHandEyeCalibrator{
private:
    std::vector<Eigen::Vector3d> robot_t_vecs_;
    std::vector<Eigen::Vector3d> robot_r_vecs_;
    std::vector<Eigen::Vector3d> robot_t_vecs_inv_;
    std::vector<Eigen::Vector3d> robot_r_vecs_inv_;
    std::vector<std::vector<Eigen::Vector3d>> pnp_t_vecs_;
    std::vector<std::vector<Eigen::Vector3d>> pnp_r_vecs_;
    std::vector< Eigen::Vector3d > pattern_pts_;
    int number_of_cameras_;
    int number_of_waypoints_;
    std::vector<std::vector<int>> cross_observation_matrix_;
    double board2ee_[6];
    double** h2e_;
    double*** cam2cam_;
    double** robot_opt_;
    double** pnp_opt_;
    std::vector<std::vector<double>> board_z_;
    std::vector<std::vector<double>> cam_z_;

    std::vector<std::vector<Eigen::Vector3d>> robot_t_rel_;
    std::vector<std::vector<Eigen::Vector3d>> robot_r_rel_;

    std::vector<std::vector<Eigen::Vector3d>> pnp_t_rel_;
    std::vector<std::vector<Eigen::Vector3d>> pnp_r_rel_;

public:
    MobileHandEyeCalibrator(const int number_of_waypoints, const int sens_quantity, const std::vector<cv::Point3f> object_points, std::vector<cv::Mat> poses, const std::vector<cv::Mat> h2e_initial_guess_vec, const cv::Mat b2ee_initial_guess, const std::vector<std::vector<int>> cross_observation_matrix,
                            std::vector<std::vector<cv::Mat>> rvec_all, std::vector<std::vector<cv::Mat>> tvec_all, std::vector<std::vector<cv::Mat>> pnp, const std::vector<std::vector<double>> tz, const std::vector<std::vector<double>> cam_z, const std::vector<std::vector<cv::Mat>> relative_robot_poses, const std::vector<std::vector<cv::Mat>> relative_cam_poses);
    ~MobileHandEyeCalibrator() {
        delete[] h2e_;
        delete[] cam2cam_;
    }
    void mobileCalibration(const std::vector<CameraInfo> camera_info, std::vector<std::vector<std::vector<cv::Point2f>>> corners, std::vector<cv::Mat> &optimal_h2e, cv::Mat &optimal_b2ee, std::vector<std::vector<cv::Mat>> &optimal_cam2cam, std::vector<cv::Mat>& optim_poses_collected, std::vector<std::vector<cv::Mat>>& optim_pnp_collected);
    void mobileJointCalibration(const std::vector<CameraInfo> camera_info, std::vector<std::vector<std::vector<cv::Point2f>>> corners, std::vector<cv::Mat> &optimal_h2e, cv::Mat &optimal_b2ee, std::vector<std::vector<cv::Mat>> &optimal_cam2cam, std::vector<cv::Mat>& optim_poses_collected, std::vector<std::vector<cv::Mat>>& optim_pnp_collected);

};

#endif //METRIC_CALIBRATOR_MOBILEHANDEYECALIBRATOR_H
