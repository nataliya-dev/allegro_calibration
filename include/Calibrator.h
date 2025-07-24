//
// Created by Davide on 12/09/24.
//

#ifndef METRIC_CALIBRATOR_CALIBRATOR_H
#define METRIC_CALIBRATOR_CALIBRATOR_H

#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "Detector.h"
#include "Metrics.h"
#include "MultiHandEyeCalibrator.h"
#include "Reader.h"
#include "utils.h"

class Calibrator {
 private:
  std::string data_;
  int number_of_waypoints_;
  int start_index_;

 public:
  std::string matrixToJson(const cv::Mat& matrix, const std::string& name);

  Calibrator(std::string data_folder, int number_of_waypoints,
             int start_index) {
    data_ = data_folder;
    number_of_waypoints_ = number_of_waypoints;
    start_index_ = start_index;
  };
  void calibration();
};

#endif  // METRIC_CALIBRATOR_CALIBRATOR_H
