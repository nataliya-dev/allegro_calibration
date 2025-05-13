//
// Created by davide on 12/09/23.
//

#ifndef METRIC_CALIBRATOR_CALIBRATOR_H
#define METRIC_CALIBRATOR_CALIBRATOR_H

#include <iostream>
#include <string>
#include <chrono>

#include "Reader.h"
#include "Detector.h"
#include "SingleHandEyeCalibrator.h"
#include "MultiHandEyeCalibrator.h"
#include "MobileHandEyeCalibrator.h"
#include "utils.h"
#include "Metrics.h"

#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

class Calibrator{
private:
    std::string data_;
    int number_of_waypoints_;
    int start_index_;

public:
    Calibrator(std::string data_folder, int number_of_waypoints, int start_index){data_ = data_folder; number_of_waypoints_ = number_of_waypoints; start_index_ = start_index;};
    void calibration();
};
/*
struct OptimizationDataBayes {
    std::vector<cv::Mat> poses_A;
    std::vector<cv::Mat> poses_B;
    Eigen::Matrix4d X_fixed;
    Eigen::Matrix4d Z_fixed;
    std::vector<std::vector<int>> cross_observation;

};


class HandEyeOptimization : public bayesopt::ContinuousModel
{
public:
    HandEyeOptimization(OptimizationDataBayes* data, size_t dim, bayesopt::Parameters par)
            : bayesopt::ContinuousModel(dim, par), data_(data) {}

    double evaluateSample(const vectord& x){
        Eigen::Matrix4d X = data_->X_fixed;
        //Eigen::Matrix4d Z = data_->Z_fixed;
        // Aggiorna solo la componente tz di X e Z basata su x
        X(2, 3) = x[0];
        //Z(2, 3) = x[1];

        double error = 0.0;

        for (size_t i = 0; i < data_->poses_A.size(); ++i) {
            if (i>0) {
                if (data_->cross_observation[i][0] && data_->cross_observation[i-1][0]) {
                    Eigen::Matrix4d A = cvMatToEigen(data_->poses_A[i-1].inv()*data_->poses_A[i]);
                    Eigen::Matrix4d B = cvMatToEigen(data_->poses_B[i-1]*data_->poses_B[i].inv());

                    Eigen::Matrix4d AX = A * X;
                    Eigen::Matrix4d ZB = X * B;

                    Eigen::Matrix4d diff = AX - ZB;
                    error += diff.squaredNorm();
                }
            }
        }

        return error;
    }

protected:
    OptimizationDataBayes* data_;

};*/





#endif //METRIC_CALIBRATOR_CALIBRATOR_H
