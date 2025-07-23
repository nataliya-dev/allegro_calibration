#include <iostream>
#include <opencv2/opencv.hpp>
// g++ -o test_corners detect_pattern.cpp `pkg-config --cflags --libs opencv4`
int main() {
  cv::Mat img = cv::imread(
      "/home/nataliya/calibration/Multi-Camera-Hand-Eye-Calibration/data/hiro/"
      "camera1/image/sample_0003.jpg");
  std::vector<cv::Point2f> corners;

  // Try different sizes:
  cv::Size boardSize(6, 7);  // rows, cols

  bool found = cv::findChessboardCorners(
      img, boardSize, corners,
      cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

  std::cout << "Corners found: " << found << std::endl;

  if (found) {
    cv::drawChessboardCorners(img, boardSize, corners, found);
    cv::imshow("Corners", img);
    cv::waitKey(0);
  }

  return 0;
}
