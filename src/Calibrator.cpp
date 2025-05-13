//
// Created by davide on 12/09/23.
//

#include "Calibrator.h"

int MAX_IMAGES = 300;


void Calibrator::calibration() {

    std::cout << "##################################################" << std::endl;
    std::cout << "Start the calibration!" << std::endl;

    //------------------------------ DATA READING ------------------------------
    // Create the Calibration Info structure
    CalibrationInfo calib_info;
    

    std::string dataset = getStringAfterLastSlash(data_);
    std::cout << "Dataset: " << dataset << std::endl;

    // Read the provided data
    Reader reader(data_);
    reader.readCalibrationInfo(calib_info);
    int calibration_setup = calib_info.getCalibSetup();

    const int number_of_cameras = calib_info.getNumberOfCams();
    int total_images = countImagesInFolder(data_ + "/" + calib_info.getCamFolderPref() + "1/image");
    int number_of_waypoints;
    int start_index = 0;
    int end_index = total_images;
    if ((number_of_waypoints_ == -1 || number_of_waypoints_ > total_images) &&
        (start_index_ == -1 || start_index_ > total_images)) {
        // If number of waypoints are not given, use all images
        number_of_waypoints = total_images;
        start_index = 0;
        std::cout << "All images will be processed!" << std::endl;
    } else {
        if (start_index_ == -1 || start_index_ > total_images) {
            // Use a quantity of images equal to the quantity supplied as input
            number_of_waypoints = number_of_waypoints_;
            std::cout << number_of_waypoints << " random images will be used!" << std::endl;

            // Generate the random number
            std::random_device rd;
            std::mt19937 gen(rd());
            // Define the range for the starting index
            std::uniform_int_distribution<> distrib(0, total_images - number_of_waypoints);

            start_index = distrib(gen);                          // This will be our random starting index
            end_index = start_index + number_of_waypoints;          // Calculate the end index based on the fixed_number

            std::cout << "Images from " << start_index << " to " << end_index << std::endl;
        } else {
            // Use a quantity of images equal to the quantity supplied as input
            number_of_waypoints = number_of_waypoints_;
            start_index = start_index_;
            end_index = start_index + number_of_waypoints;          // Calculate the end index based on the fixed_number

            std::cout << "Images from " << start_index << " to " << end_index << std::endl;
        }
    }

    // Check reliability of the provided data
    checkData(data_, calib_info.getCamFolderPref(), number_of_cameras);

    // Create the Camera Network Info structure
    std::vector <CameraInfo> camera_network_info(number_of_cameras);
    for (int i = 0; i < camera_network_info.size(); i++) {
        camera_network_info[i].setParameters(data_ + "/" + calib_info.getCamFolderPref() + std::to_string(i + 1),
                                             calib_info.getResizeFactor());
    }

    // Read the image and pose collections
    std::vector <std::vector<cv::Mat>> original_poses(number_of_cameras);
    std::vector <std::vector<cv::Mat>> images_collected = reader.readImages(number_of_cameras,
                                                                            calib_info.getResizeFactor(), start_index,
                                                                            end_index);
    std::vector <std::vector<cv::Mat>> poses_collected = reader.readRobotPoses(number_of_cameras, original_poses,
                                                                               start_index, end_index);



    // ------------------------------ DATA COLLECTION ------------------------------
    // Initialize the vectors of correct images, poses, corners and the cross observation matrix
    std::vector <std::vector<cv::Mat>> correct_images(number_of_cameras), correct_images_single(
            number_of_cameras), correct_poses(number_of_cameras), correct_poses_single(number_of_cameras);
    std::vector < std::vector < std::vector < cv::Point2f>>> correct_corners_only(
            number_of_cameras), correct_corners_single_only(number_of_cameras), correct_corners(number_of_waypoints,
                                                                                                std::vector <
                                                                                                std::vector <
                                                                                                cv::Point2f >> (
                                                                                                        number_of_cameras)), correct_corners_single(
            number_of_waypoints, std::vector < std::vector < cv::Point2f >> (number_of_cameras));
    std::vector <std::vector<int>> cross_observation_matrix(number_of_waypoints, std::vector<int>(number_of_cameras,
                                                                                                  0)), cross_observation_matrix_single(
            number_of_waypoints, std::vector<int>(number_of_cameras, 0));
    std::vector <std::vector<cv::Mat>> rvec_all(number_of_waypoints, std::vector<cv::Mat>(number_of_cameras)), tvec_all(
            number_of_waypoints, std::vector<cv::Mat>(number_of_cameras)), rvec_used(number_of_cameras,
                                                                                     std::vector<cv::Mat>(
                                                                                             number_of_waypoints)), tvec_used(
            number_of_cameras, std::vector<cv::Mat>(number_of_waypoints));

    // Detect the calibration pattern
    Detector detector(calib_info, camera_network_info, number_of_waypoints);
    detector.patternDetection(images_collected, poses_collected, correct_images, correct_poses, correct_corners_only,
                              correct_corners, cross_observation_matrix, rvec_all, tvec_all);

    std::vector<int> selected_poses = selectPoses(cross_observation_matrix, MAX_IMAGES);
    for (int cam = 0; cam < number_of_cameras; cam++){
        int counter_poses = 0;
        std::cout << "########## " << cam << " ##########" << std::endl;
        for (int wp = 0; wp < number_of_waypoints; wp++){
            if (cross_observation_matrix[wp][cam] && selected_poses[wp]){
                counter_poses++;
                //std::cout << "Pose " << wp << std::endl;
            }
        }
        std::cout << "Camera " << cam << " has " << counter_poses << " poses" << std::endl;
    }
    int multi_camera_poses = countMultiCameraPoses(selected_poses, cross_observation_matrix);

    std::cout << "Numero di pose viste da più camere: " << multi_camera_poses << std::endl;


    // Convert robot poses with double values
    for (int j = 0; j < number_of_cameras; j++) {
        for (int i = 0; i < poses_collected[0].size(); i++) {
            poses_collected[j][i].convertTo(poses_collected[j][i], CV_64F);
        }
    }


    std::vector <std::vector<cv::Mat>> rototras_vec(number_of_cameras), rototras_all(number_of_cameras);
    for (int i = 0; i < number_of_cameras; i++) {
        std::vector <cv::Mat> rototras_camera(number_of_waypoints);
        std::vector <cv::Mat> rototras_camera_push;
        for (int j = 0; j < number_of_waypoints; j++) {
            if (cross_observation_matrix[j][i]) {
                cv::Mat rototras;
                cv::Mat rotation;
                cv::Rodrigues(rvec_all[j][i], rotation);
                getRotoTras<double>(rotation, tvec_all[j][i], rototras);

                getTras<double>(rototras, tvec_used[i][j]);
                cv::Mat rotation_temp;
                getRoto<double>(rototras, rotation_temp);
                cv::Rodrigues(rotation_temp, rvec_used[i][j]);

                rototras_camera[j] = rototras;
                rototras_camera_push.push_back(rototras);
            }
        }
        rototras_vec[i] = rototras_camera;
        rototras_all[i] = rototras_camera_push;
    }






    // RELATIVE POSES
    std::vector <std::vector<cv::Mat>> relative_robot_poses(number_of_cameras,
                                                            std::vector<cv::Mat>(number_of_waypoints - 1));
    for (int i = 0; i < number_of_cameras; i++) {
        for (int j = 0; j < number_of_waypoints - 1; j++) {
            relative_robot_poses[i][j] = poses_collected[i][j].inv() * poses_collected[i][j + 1];
        }
    }

    std::vector <std::vector<cv::Mat>> relative_cam_poses(number_of_cameras,
                                                          std::vector<cv::Mat>(number_of_waypoints - 1));
    for (int i = 0; i < number_of_cameras; i++) {
        for (int j = 0; j < number_of_waypoints - 1; j++) {
            if (cross_observation_matrix[j][i] && cross_observation_matrix[j + 1][i]) {
                relative_cam_poses[i][j] = rototras_vec[i][j] * rototras_vec[i][j + 1].inv();
            }
        }
    }


    // ---------------------------------- SVD Elaboration ----------------------------------
    std::vector <std::vector<double>> d_board_whole(number_of_cameras), ang_board_whole(number_of_cameras);
    std::vector <std::vector<double>> d_camera_whole(number_of_cameras), ang_camera_whole(number_of_cameras);

    std::seed_seq seed{0};
    std::random_device rd;
    std::mt19937 gen(seed);
    double std_dev_tras = 0.02;
    double std_dev_rot = 0.10;
    std::normal_distribution<> d_tras(0, std_dev_tras); // Gaussian distribution with 0 mean and std_dev standard deviation
    std::normal_distribution<> d_rot(0, std_dev_rot);

    for (int cam = 0; cam < number_of_cameras; cam++) {
        std::vector <cv::Mat> robot_perturbation;
        bool data_perturbation_function = false;
        std::cout << "CAMERA PERTURBATED " << std::to_string(cam+1) << std::endl;

        if (data_perturbation_function) {
            data_perturbation(relative_robot_poses[cam], std_dev_tras, std_dev_rot, d_tras, d_rot, gen);
        }
        /*if (data_perturbation_function) {
            data_perturbation_camera(relative_cam_poses[cam], std_dev_tras, std_dev_rot, cross_observation_matrix, cam);
        }*/
    }





    bool RAL_CALIB = true;
    if (RAL_CALIB) {
        for (int i = 0; i < number_of_cameras; i++) {
            std::ofstream outFileRob(data_ + "/robot" + std::to_string(i + 1) + ".txt");
            std::ofstream outFileCam(data_ + "/cam" + std::to_string(i + 1) + ".txt");
            std::ofstream outFileRob3d(data_ + "/robot3d" + std::to_string(i + 1) + ".txt");
            std::ofstream outFileCam3d(data_ + "/cam3d" + std::to_string(i + 1) + ".txt");

            for (int j = 0; j < number_of_waypoints - 1; j++) {
                if (cross_observation_matrix[j][0] && cross_observation_matrix[j + 1][0] &&
                    cross_observation_matrix[j][1] && cross_observation_matrix[j + 1][1] &&
                    cross_observation_matrix[j][2] && cross_observation_matrix[j + 1][2]) {
                    cv::Mat temp = (cv::Mat_<double>(4, 4) << 0, 0, 1, 0,
                            -1, 0, 0, 0,
                            0, -1, 0, 0);
                    cv::Mat translation_rob, rotation_rob, translation_cam, rotation_cam;
                    getTras<double>(relative_robot_poses[i][j], translation_rob);
                    getRoto<double>(relative_robot_poses[i][j], rotation_rob);
                    getTras<double>(temp*relative_cam_poses[i][j], translation_cam);
                    getRoto<double>(temp*relative_cam_poses[i][j], rotation_cam);
                    cv::Mat euler_rob = rotationMatrixToEulerAngles<double>(rotation_rob);
                    cv::Mat euler_cam = rotationMatrixToEulerAngles<double>(rotation_cam);
                    cv::Vec4d quat_rob = rotationMatrixToQuaternion(rotation_rob);
                    cv::Vec4d quat_cam = rotationMatrixToQuaternion(rotation_cam);
                    if (outFileRob.is_open()) {
                        outFileRob << translation_rob.at<double>(0) << " " << translation_rob.at<double>(1) << " "
                                   << euler_rob.at<double>(2) << std::endl;
                    } else {
                        std::cerr << "Unable to open file for writing." << std::endl;
                    }
                    if (outFileCam.is_open()) {
                        outFileCam << translation_cam.at<double>(0) << " " << translation_cam.at<double>(1) << " "
                                   << euler_cam.at<double>(2) << std::endl;
                    } else {
                        std::cerr << "Unable to open file for writing." << std::endl;
                    }


                    if (outFileRob3d.is_open()) {
                        outFileRob3d << translation_rob.at<double>(0) << " " << translation_rob.at<double>(1) << " " << translation_rob.at<double>(2) << " "
                                << quat_rob[0] << " " << quat_rob[1] << " " << quat_rob[2] << " " << quat_rob[3] << std::endl;
                    } else {
                        std::cerr << "Unable to open file for writing." << std::endl;
                    }
                    if (outFileCam3d.is_open()) {
                        outFileCam3d << translation_cam.at<double>(0) << " " << translation_cam.at<double>(1) << " " << translation_cam.at<double>(2) << " "
                                   << quat_cam[0] << " " << quat_cam[1] << " " << quat_cam[2] << " " << quat_cam[3] << std::endl;
                    } else {
                        std::cerr << "Unable to open file for writing." << std::endl;
                    }
                }
            }

            outFileRob.close();
            outFileCam.close();
            outFileRob3d.close();
            outFileCam3d.close();
        }
    }

    if (calib_info.getCalibSetup() == 2) {
        std::vector <cv::Mat> svd_mat_vec(number_of_cameras);
        std::vector <cv::Mat> svd_mat_inv_vec(number_of_cameras);
        for (int i = 0; i < number_of_cameras; i++) {
            cv::Mat data_pnp(rototras_all[i].size(), 3, CV_64F);
            cv::Mat data_robot(rototras_all[i].size(), 3, CV_64F);

            cv::Mat data_pnp_inv(rototras_all[i].size(), 3, CV_64F);
            cv::Mat data_robot_inv(rototras_all[i].size(), 3, CV_64F);

            // Translation matrices extraction
            transVec2mat(rototras_vec[i], cross_observation_matrix, data_pnp, true, i);
            transVec2mat(poses_collected[i], cross_observation_matrix, data_robot, false, i);
            transVec2mat(rototras_vec[i], cross_observation_matrix, data_pnp_inv, false, i);
            transVec2mat(poses_collected[i], cross_observation_matrix, data_robot_inv, true, i);

            // Get SVD
            cv::Mat svd_mat = getSVD(data_robot, data_pnp);
            std::cout << "SVD: " << svd_mat << std::endl;
            svd_mat_vec[i] = svd_mat;
            //cv::Mat svd_mat_inv = getSVD(data_robot_inv, data_pnp_inv);
            //std::cout << "SVD inv: " << svd_mat_inv << std::endl;
            //svd_mat_inv_vec[i] = svd_mat_inv;


            // DAVIDE
            cv::Mat svd_temp = (cv::Mat_<double>(4, 4) << svd_mat_vec[i].at<double>(0, 0), svd_mat_vec[i].at<double>(0,
                                                                                                                     1), svd_mat_vec[i].at<double>(
                    0, 2), 0,
                    svd_mat_vec[i].at<double>(1, 0), svd_mat_vec[i].at<double>(1, 1), svd_mat_vec[i].at<double>(1,
                                                                                                                2), 0,
                    svd_mat_vec[i].at<double>(2, 0), svd_mat_vec[i].at<double>(2, 1), svd_mat_vec[i].at<double>(2,
                                                                                                                2), 0,
                    0, 0, 0, 1);
            for (int j = 0; j < number_of_waypoints; j++) {
                if (cross_observation_matrix[j][i]) {
                    svd_mat_inv_vec[i] = poses_collected[i][j].inv() * svd_temp * rototras_vec[i][j].inv();
                    std::cout << "Esimated svd inv: " << svd_mat_inv_vec[i] << std::endl;
                    break;
                }
            }
        }


        // ------------------ Feature extraction and feature matching ------------------
        std::vector <std::vector<cv::Mat>> rvec(number_of_cameras, std::vector<cv::Mat>(number_of_waypoints)), tvec(
                number_of_cameras, std::vector<cv::Mat>(number_of_waypoints));

        for (int i = 0; i < number_of_cameras; i++) {
            for (int j = 0; j < number_of_waypoints; j++) {
                if (cross_observation_matrix[j][i]) {
                    rvec[i][j] = rvec_all[j][i];
                    tvec[i][j] = tvec_all[j][i];
                }
            }
        }

        std::vector < std::vector < std::vector < cv::Point3d>>> outputPoints_whole(number_of_cameras,
                                                                                    std::vector < std::vector <
                                                                                    cv::Point3d >> (
                                                                                            images_collected[0].size() -
                                                                                            1));
        std::vector<std::vector<std::vector<cv::KeyPoint>>> keypoints1_whole(number_of_cameras, std::vector<std::vector<cv::KeyPoint>>(number_of_waypoints));
        std::vector<std::vector<std::vector<cv::KeyPoint>>> keypoints2_whole(number_of_cameras, std::vector<std::vector<cv::KeyPoint>>(number_of_waypoints));

        std::vector <std::vector<double>> alignment_score(number_of_cameras);
        int rgbd_sensor = 0;
        if (!rgbd_sensor) {
            for (int j = 0; j < number_of_cameras; j++) {
                std::cout << "################## Camera " << std::to_string(j + 1) << " ##################"
                          << std::endl;
                cv::Mat svd_RC_complete = (cv::Mat_<double>(4, 4)
                        << svd_mat_inv_vec[j].at<double>(0, 0), svd_mat_inv_vec[j].at<double>(0,
                                                                                              1), svd_mat_inv_vec[j].at<double>(
                        0, 2), 0,
                        svd_mat_inv_vec[j].at<double>(1, 0), svd_mat_inv_vec[j].at<double>(1,
                                                                                           1), svd_mat_inv_vec[j].at<double>(
                        1, 2), 0,
                        svd_mat_inv_vec[j].at<double>(2, 0), svd_mat_inv_vec[j].at<double>(2,
                                                                                           1), svd_mat_inv_vec[j].at<double>(
                        2, 2), 0,
                        0, 0, 0, 1);

                cv::Mat svd_WB_complete = (cv::Mat_<double>(4, 4)
                        << svd_mat_vec[j].at<double>(0, 0), svd_mat_vec[j].at<double>(0, 1), svd_mat_vec[j].at<double>(
                        0,
                        2), 0,
                        svd_mat_vec[j].at<double>(1, 0), svd_mat_vec[j].at<double>(1, 1), svd_mat_vec[j].at<double>(1,
                                                                                                                    2), 0,
                        svd_mat_vec[j].at<double>(2, 0), svd_mat_vec[j].at<double>(2, 1), svd_mat_vec[j].at<double>(2,
                                                                                                                    2), 0,
                        0, 0, 0, 1);

                // Extract feature and matches with Deep Image Matching
                std::vector <std::vector<cv::KeyPoint>> keypoints1_vec(number_of_waypoints), keypoints2_vec(
                        number_of_waypoints);
                //extractAndMatchSIFT(images_collected[0], keypoints1_vec, keypoints2_vec);

                findDeepMatches(data_ + "/camera" + std::to_string(j + 1) + "/matches/", images_collected[j],
                                keypoints1_vec, keypoints2_vec, start_index, end_index, calib_info.getResizeFactor());

                // Check if vec1 is not empty before trying to erase the first element
                if (!keypoints2_vec.empty()) {
                    keypoints2_vec.erase(keypoints2_vec.begin());
                }

                // Check if vec2 is not empty before trying to remove the last element
                if (!keypoints1_vec.empty()) {
                    keypoints1_vec.pop_back();
                }
                keypoints1_whole[j] = keypoints1_vec;
                keypoints2_whole[j] = keypoints2_vec;


                std::vector <std::vector<cv::Point3d>> outputPoints_vec(images_collected[j].size() - 1);
                for (int i = 0; i < images_collected[j].size() - 1; i++) {
                    if (cross_observation_matrix[i][j] && cross_observation_matrix[i + 1][j] && keypoints1_vec[i].size()>0) {
                        triangulatePoints(rvec[j][i], tvec[j][i], rvec[j][i + 1], tvec[j][i + 1], keypoints1_vec[i],
                                          keypoints2_vec[i], camera_network_info[j].getCameraMatrix(),
                                          outputPoints_vec[i]);
                    }
                }


                outputPoints_whole[j] = outputPoints_vec;
                /*for (int i = 0; i < outputPoints_vec.size(); i++){
                    plotWithPCL(outputPoints_vec[i], svd_mat_vec[j]);
                }*/

                std::vector<double> trajectory_aligned(keypoints1_vec.size());

            }
        }



        std::vector<std::vector<double>> board_distance_whole(number_of_cameras, std::vector<double> (number_of_waypoints-1));
        for (int i = 0; i < number_of_cameras; i++) {
            std::cout << "################## Camera " << std::to_string(i + 1) << " ##################" << std::endl;
            cv::Mat svd_RC_complete = (cv::Mat_<double>(4, 4)
                    << svd_mat_inv_vec[i].at<double>(0, 0), svd_mat_inv_vec[i].at<double>(0,
                                                                                          1), svd_mat_inv_vec[i].at<double>(
                    0, 2), 0,
                    svd_mat_inv_vec[i].at<double>(1, 0), svd_mat_inv_vec[i].at<double>(1,
                                                                                       1), svd_mat_inv_vec[i].at<double>(
                    1, 2), 0,
                    svd_mat_inv_vec[i].at<double>(2, 0), svd_mat_inv_vec[i].at<double>(2,
                                                                                       1), svd_mat_inv_vec[i].at<double>(
                    2, 2), 0,
                    0, 0, 0, 1);

            cv::Mat svd_WB_complete = (cv::Mat_<double>(4, 4)
                    << svd_mat_vec[i].at<double>(0, 0), svd_mat_vec[i].at<double>(
                    0, 1), svd_mat_vec[i].at<double>(0, 2), 0,
                    svd_mat_vec[i].at<double>(1, 0), svd_mat_vec[i].at<double>(1, 1), svd_mat_vec[i].at<double>(1,
                                                                                                                2), 0,
                    svd_mat_vec[i].at<double>(2, 0), svd_mat_vec[i].at<double>(2, 1), svd_mat_vec[i].at<double>(2,
                                                                                                                2), 0,
                    0, 0, 0, 1);


            std::vector<double> z_robot(number_of_waypoints - 1);
            std::vector<double> z_pnp(number_of_waypoints - 1);
            //std::vector<double> z_ess(number_of_waypoints - 1);
            std::vector<double> z_robot_copy(number_of_waypoints - 1);
            std::vector<double> z_pnp_copy(number_of_waypoints - 1);
            //std::vector<double> z_ess_copy(number_of_waypoints - 1);

            std::vector <cv::Mat> robot3d(number_of_waypoints - 1);
            std::vector <cv::Mat> cam3d(number_of_waypoints - 1);
            std::vector <cv::Mat> rel_robot3d(number_of_waypoints - 1);
            std::vector <cv::Mat> rel_cam3d(number_of_waypoints - 1);

            std::vector <cv::Mat> robot2d(number_of_waypoints - 1);
            std::vector <cv::Mat> cam2d(number_of_waypoints - 1);
            std::vector <cv::Mat> rel_robot2d(number_of_waypoints - 1);
            std::vector <cv::Mat> rel_cam2d(number_of_waypoints - 1);

            std::vector <cv::Mat> relative_cam_vec(number_of_waypoints - 1);

            for (int j = 0; j < number_of_waypoints - 1; j++) {
                if (cross_observation_matrix[j][i] && cross_observation_matrix[j + 1][i]) {
                    cv::Mat relative_robot_pose = relative_robot_poses[i][j];
                    cv::Mat relative_camera_pose =
                            svd_RC_complete * relative_cam_poses[i][j]* svd_RC_complete.inv();
                    relative_cam_vec[j] = relative_camera_pose;
                    cv::Mat tras_robot_abs, tras_pnp_abs;
                    getTras<double>(poses_collected[i][j], tras_robot_abs);
                    getTras<double>(svd_WB_complete * rototras_vec[i][j].inv(), tras_pnp_abs);
                    robot3d[j] = (tras_robot_abs);
                    cam3d[j] = (tras_pnp_abs);
                    cv::Mat translation_board;
                    getTras<double>(svd_RC_complete * rototras_vec[i][j], translation_board);
                    double x = translation_board.at<double>(0); // Access the first element
                    double y = translation_board.at<double>(1); // Access the second element
                    double board_distance = std::sqrt(x*x + y*y);
                    board_distance_whole[i][j] = board_distance;
                    cv::Mat mask, R, t;
                    std::vector<cv::Point2f> points1, points2;
                    convertKeypointsToPoint2f(keypoints1_whole[i][j], points1);
                    convertKeypointsToPoint2f(keypoints2_whole[i][j], points2);
                    /*cv::Mat essentialMatrix = findEssentialMat(points1, points2,
                                                               camera_network_info[i].getCameraMatrix(), cv::RANSAC);
                    recoverPose(essentialMatrix, points1, points2, camera_network_info[i].getCameraMatrix(), R, t,
                                mask);


                    std::cout << "######### Pose " << std::to_string(j) << " to " << std::to_string(j + 1)
                              << " ###########"
                              << std::endl;

                    cv::Mat svd_rot;
                    getRoto<double>(svd_mat_inv_vec[i], svd_rot);
                    cv::Mat rotation_ess = svd_rot*R.inv()*svd_rot.inv();*/


                    cv::Mat rotation_robot, rotation_pnp;
                    getRoto<double>(relative_robot_pose, rotation_robot);
                    getRoto<double>(relative_camera_pose, rotation_pnp);
                    cv::Mat tras_robot, tras_pnp;
                    getTras<double>(relative_robot_pose, tras_robot);
                    getTras<double>(relative_camera_pose, tras_pnp);
                    //std::cout << "Rotation ess: " << rotation_ess << std::endl;
                    //std::cout << "Rotation pnp: " << rotation_pnp << std::endl;


                    cv::Mat euler_robot = rotationMatrixToEulerAngles<double>(rotation_robot);
                    cv::Mat euler_pnp = rotationMatrixToEulerAngles<double>(rotation_pnp);
                    //cv::Mat euler_ess = rotationMatrixToEulerAngles<double>(rotation_ess);
                    rel_robot3d[j] = (tras_robot);
                    rel_cam3d[j] = (tras_pnp);
                    rel_robot2d[j] = ((cv::Mat_<double>(1, 2) << tras_robot.at<double>(0), tras_robot.at<double>(1)));
                    rel_cam2d[j] = ((cv::Mat_<double>(1, 2) << tras_pnp.at<double>(0), tras_pnp.at<double>(1)));
                    z_robot[j] = (euler_robot.at<double>(2));
                    z_pnp[j] = (euler_pnp.at<double>(2));

                    //z_ess[j] = (euler_ess.at<double>(2));
                }
            }


            //plot3DTrajectories(robot3d, cam3d);
            //plot3DTrajectories(rel_robot3d, rel_cam3d);

            LinearRegressionResult result = simpleLinearRegression(z_robot, z_pnp);
            //LinearRegressionResult result_ess = simpleLinearRegression(z_robot, z_ess);
            // Plot the result
            plotLinearRegression(z_robot, z_pnp, result);
            //plotLinearRegression(z_robot, z_ess, result_ess);
            std::cout << "Slope: " << result.slope << ", Intercept: " << result.intercept << std::endl;
            //std::cout << "Slope ess: " << result_ess.slope << ", Intercept: " << result_ess.intercept << std::endl;

            z_robot_copy = z_robot;
            z_pnp_copy = z_pnp;
            //z_ess_copy = z_ess;

            // Remove points far from the regression line
            double threshold = 0.001;
            std::vector<int> removedIndices = removeOutliersFromLine(z_robot_copy, z_pnp_copy, result, threshold);
            // Display removed indices
            std::cout << "Indices of removed points: ";
            for (int idx: removedIndices) {
                std::cout << idx << " ";
            }

            std::cout << std::endl;
            for (int idx: removedIndices) {
                if (cross_observation_matrix[idx][i] && cross_observation_matrix[idx + 1][i]) {

                    cv::Mat correct_roto, init_robot_roto;
                    getRoto<double>(relative_cam_vec[idx], correct_roto);
                    getRoto<double>(relative_robot_poses[i][idx], init_robot_roto);
                    //z_robot[idx] = (z_pnp[idx] - result.intercept) / result.slope;
                    z_robot[idx] = z_pnp[idx];
                    cv::Mat euler_robot = rotationMatrixToEulerAngles<double>(init_robot_roto);
                    euler_robot.at<double>(2) = z_robot[idx];
                    cv::Mat roto_transformed = eulerAnglesToRotationMatrix<double>(euler_robot);
                    roto_transformed.copyTo(relative_robot_poses[i][idx](
                            cv::Rect(0, 0, 3, 3))); // Copy 3x3 rotation into the top-left of the 4x4 matrix
                    //cross_observation_matrix[idx][i] = 0;
                }
            }

            // Plot the result
            LinearRegressionResult result2 = simpleLinearRegression(z_robot, z_pnp);

            plotLinearRegression(z_robot, z_pnp, result2);
            std::cout << "Slope: " << result2.slope << ", Intercept: " << result2.intercept << std::endl;
        }


        std::vector <std::vector<cv::Mat>> pointcloud_whole(number_of_cameras,
                                                            std::vector<cv::Mat>(number_of_waypoints));


        // POINTCLOUD ELABORATION
        double ground_plane_confidence = 0.9999;
        //double ground_plane_confidence = 0.95;
        for (int cam = 0; cam < number_of_cameras; cam++) {
            std::vector <cv::Mat> pointcloud_vec(number_of_waypoints);
            int counter = 0;
            std::vector<double> d_board_vec(number_of_waypoints), ang_board_vec(number_of_waypoints);
            std::vector<double> d_camera_vec(number_of_waypoints), ang_camera_vec(number_of_waypoints);

            std::cout << "#################################### CAMERA " << std::to_string(cam + 1)
                      << "####################################" << std::endl;
            // ------------------------------------------ Ground plane generation ------------------------------------------

            if (rgbd_sensor) {
                namespace fs = std::filesystem;
                std::vector <std::vector<fs::directory_entry>> entries(number_of_cameras);
                for (const auto &entry: fs::directory_iterator(
                        data_ + "/camera" + std::to_string(cam + 1) + "/clouds/")) {
                    if (fs::is_regular_file(entry)) {
                        // Check if the file is an image (you can add more extensions as needed)
                        std::string fileExtension = entry.path().extension();
                        if (fileExtension == ".txt") {
                            entries[cam].push_back(entry);
                        }
                    }
                }

                // Sort the filenames in ascending order
                std::sort(entries[cam].begin(), entries[cam].end(), compareFilenames);

                /*for (const auto &entry: entries[cam]) {
                    cv::Mat temp_pcl = readPointsFromFile(entry.path().string());
                    cv::Mat newMat;
                    cv::Mat rowToAdd = cv::Mat::ones(1, temp_pcl.rows, temp_pcl.type());
                    cv::vconcat(temp_pcl.t(), rowToAdd, newMat);

                    pointcloud_vec[counter] = newMat;
                    counter += 1;
                }*/


                int min = end_index <= entries[cam].size() ? end_index : entries[cam].size();
                for (int i = start_index; i < min; i++) {
                    const auto &entry = entries[cam][i];
                    cv::Mat temp_pcl = readPointsFromFile(entry.path().string());
                    if (!temp_pcl.empty()) {
                        cv::Mat newMat;
                        cv::Mat rowToAdd = cv::Mat::ones(1, temp_pcl.rows, temp_pcl.type());
                        cv::vconcat(temp_pcl.t(), rowToAdd, newMat);

                        pointcloud_vec[i] = newMat;
                        //counter += 1;
                    }
                }


                for (int i = 0; i < pointcloud_vec.size(); i++) {
                    if (cross_observation_matrix[i][cam] && !pointcloud_vec[i].empty()) {
                        cv::Mat pointcloud_board = rototras_vec[cam][i].inv() * pointcloud_vec[i];
                        cv::Mat null_tras = (cv::Mat_<double>(3, 1) << 0, 0, 0);
                        cv::Mat svd;
                        getRotoTras<double>(svd_mat_vec[cam], null_tras, svd);
                        cv::Mat pointcloud_world = svd * pointcloud_board;

                        //pointcloud_world = pointcloud_vec[i];

                        // If read from triangulation

                        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud <pcl::PointXYZ>);

                        // Fill the cloud with some points
                        for (int k = 0; k < pointcloud_board.cols; k++) {
                            cloud->points.push_back(
                                    pcl::PointXYZ(pointcloud_world.at<double>(0, k), pointcloud_world.at<double>(1, k),
                                                  pointcloud_world.at<double>(2, k)));
                            cloud->width = cloud->points.size();
                            cloud->height = 1;
                            cloud->is_dense = false;
                        }

                        // Create a PCLVisualizer object
                        /*pcl::visualization::PCLVisualizer viewer("3D Viewer");
                        viewer.setBackgroundColor(0, 0, 0);
                        viewer.addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
                        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3,
                                                                "sample cloud");
                        viewer.addCoordinateSystem(1.0);
                        viewer.initCameraParameters();

                        // Main visualization loop
                        while (!viewer.wasStopped()) {
                            viewer.spinOnce(100);
                        }*/


                        // RANSAC
                        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
                        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
                        // Create the segmentation object
                        pcl::SACSegmentation <pcl::PointXYZ> seg;
                        // Optional
                        seg.setOptimizeCoefficients(true);
                        // Mandatory
                        seg.setModelType(pcl::SACMODEL_PLANE);
                        seg.setMethodType(pcl::SAC_RANSAC);


                        //seg.setDistanceThreshold (0.01);
                        seg.setDistanceThreshold(1-ground_plane_confidence);
                        seg.setInputCloud(cloud);
                        seg.segment(*inliers, *coefficients);

                        pcl::ExtractIndices <pcl::PointXYZ> extract;
                        pcl::PointCloud<pcl::PointXYZ>::Ptr inliersCloud(new pcl::PointCloud <pcl::PointXYZ>);
                        extract.setInputCloud(cloud);
                        extract.setIndices(inliers);
                        extract.filter(*inliersCloud);


                        if (inliers->indices.size() == 0) {
                            PCL_ERROR("Could not estimate a planar model for the given dataset.\n");
                        }
                        /*std::cerr << "Model coefficients: " << coefficients->values[0] << " "
                                  << coefficients->values[1] << " "
                                  << coefficients->values[2] << " "
                                  << coefficients->values[3] << std::endl;*/

                        cv::Mat normal_wrt_world = (cv::Mat_<double>(3, 1)
                                << coefficients->values[0], coefficients->values[1], coefficients->values[2]);
                        //cv::Mat normal_wrt_world = svd_mat*normal_wrt_board;
                        cv::Point3d normal_vec_wrt_world(normal_wrt_world.at<double>(0), normal_wrt_world.at<double>(1),
                                                         normal_wrt_world.at<double>(2));

                        cv::Point3d normal_world(0, 0, 1);
                        double scalar =
                                normal_world.x * normal_vec_wrt_world.x + normal_world.y * normal_vec_wrt_world.y +
                                normal_world.z * normal_vec_wrt_world.z;
                        /*pcl::visualization::PCLVisualizer::Ptr viewer2(new pcl::visualization::PCLVisualizer("3D Viewer PLANE"));
                        viewer2->setBackgroundColor(0, 0, 0);

                        // Add the original point cloud in white
                        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloudColorHandler(cloud, 255, 255, 255);
                        viewer2->addPointCloud(cloud, cloudColorHandler, "originalCloud");

                        // Add the inliers in red
                        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> inliersColorHandler(inliersCloud, 255, 0, 0);
                        viewer2->addPointCloud(inliersCloud, inliersColorHandler, "inliersCloud");
                        viewer2->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "inliersCloud");

                        // Add coordinate system and initialize camera view
                        viewer2->addCoordinateSystem(1.0);
                        viewer2->initCameraParameters();

                        // Main visualization loop
                        while (!viewer2->wasStopped()) {
                            viewer2->spinOnce(100);
                        }*/

                        d_board_vec[i] = coefficients->values[3];
                        ang_board_vec[i] = scalar;
                    }
                }


                for (int i = 0; i < number_of_waypoints; i++) {
                    if (cross_observation_matrix[i][0] && !pointcloud_vec[i].empty()) {

                        cv::Mat null_tras = (cv::Mat_<double>(3, 1) << 0, 0, 0);
                        cv::Mat svd;
                        getRotoTras<double>(svd_mat_inv_vec[cam], null_tras, svd);
                        cv::Mat pointcloud_world = svd * pointcloud_vec[i];

                        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud <pcl::PointXYZ>);

                        // Fill the cloud with some points
                        for (int k = 0; k < pointcloud_world.cols; k++) {
                            cloud->points.push_back(
                                    pcl::PointXYZ(pointcloud_world.at<double>(0, k), pointcloud_world.at<double>(1, k),
                                                  pointcloud_world.at<double>(2, k)));
                            cloud->width = cloud->points.size();
                            cloud->height = 1;
                            cloud->is_dense = false;
                        }

                        // Create a PCLVisualizer object
                        /*pcl::visualization::PCLVisualizer viewer("3D Viewer");
                        viewer.setBackgroundColor(0, 0, 0);
                        viewer.addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
                        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10,
                                                                "sample cloud");
                        viewer.addCoordinateSystem(1.0);
                        viewer.initCameraParameters();*/

                        // RANSAC
                        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
                        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
                        // Create the segmentation object
                        pcl::SACSegmentation <pcl::PointXYZ> seg;
                        // Optional
                        seg.setOptimizeCoefficients(true);
                        // Mandatory
                        seg.setModelType(pcl::SACMODEL_PLANE);
                        seg.setMethodType(pcl::SAC_RANSAC);
                        seg.setDistanceThreshold(1-ground_plane_confidence);
                        seg.setInputCloud(cloud);
                        seg.segment(*inliers, *coefficients);

                        pcl::ExtractIndices <pcl::PointXYZ> extract;
                        pcl::PointCloud<pcl::PointXYZ>::Ptr inliersCloud(new pcl::PointCloud <pcl::PointXYZ>);
                        extract.setInputCloud(cloud);
                        extract.setIndices(inliers);
                        extract.filter(*inliersCloud);


                        if (inliers->indices.size() == 0) {
                            PCL_ERROR("Could not estimate a planar model for the given dataset.\n");
                        }





                        std::ofstream csvFile(data_ + "/ground_plane_points" + std::to_string(cam+1)+ ".csv");

                        // Check if the file is open
                        if (csvFile.is_open()) {

                            // Iterate through the points in the inliers cloud
                            for (size_t index = 0; index < inliersCloud->points.size(); ++index) {

                                // Extract the original point
                                // Get the point
                                pcl::PointXYZ point = inliersCloud->points[index];

                                // Write the point's coordinates to the file, separated by commas
                                csvFile << point.x << "," << point.y << "," << point.z << "\n";
                            }

                            // Close the file
                            csvFile.close();
                        } else {
                            std::cerr << "Unable to open file for writing.\n";
                        }






                        /*std::cerr << "Model coefficients: " << coefficients->values[0] << " "
                                  << coefficients->values[1] << " "
                                  << coefficients->values[2] << " "
                                  << coefficients->values[3] << std::endl;*/
                            /*pcl::visualization::PCLVisualizer::Ptr viewer2(
                                    new pcl::visualization::PCLVisualizer("3D Viewer PLANE"));
                            viewer2->setBackgroundColor(1, 1, 1);

                            // Add the original point cloud in white
                            pcl::visualization::PointCloudColorHandlerCustom <pcl::PointXYZ> cloudColorHandler(cloud,
                                                                                                               255, 255,
                                                                                                               255);
                            viewer2->addPointCloud(cloud, cloudColorHandler, "originalCloud");

                            // Add the inliers in red
                            pcl::visualization::PointCloudColorHandlerCustom <pcl::PointXYZ> inliersColorHandler(
                                    inliersCloud, 255, 0, 0);
                            viewer2->addPointCloud(inliersCloud, inliersColorHandler, "inliersCloud");
                            viewer2->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1,
                                                                      "inliersCloud");

                            // Add coordinate system and initialize camera view
                            viewer2->addCoordinateSystem(1.0);
                            viewer2->initCameraParameters();

                            // Main visualization loop
                            while (!viewer2->wasStopped()) {
                                viewer2->spinOnce(100);
                            }*/

                        cv::Mat normal_wrt_world = (cv::Mat_<double>(3, 1)
                                << coefficients->values[0], coefficients->values[1], coefficients->values[2]);
                        cv::Point3d normal_vec_wrt_world(normal_wrt_world.at<double>(0), normal_wrt_world.at<double>(1),
                                                         normal_wrt_world.at<double>(2));
                        cv::Point3d normal_world(0, 0, 1);
                        double scalar =
                                normal_world.x * normal_vec_wrt_world.x + normal_world.y * normal_vec_wrt_world.y +
                                normal_world.z * normal_vec_wrt_world.z;

                        d_camera_vec[i] = coefficients->values[3];
                        ang_camera_vec[i] = scalar;
                    }
                }
            } else {
                for (int i = 0; i < pointcloud_vec.size() - 1; i++) {
                    if (cross_observation_matrix[i][cam] && cross_observation_matrix[i + 1][cam]) {
                        cv::Mat temp_pcl(outputPoints_whole[cam][i].size(), 3, CV_64F);
                        for (size_t j = 0; j < outputPoints_whole[cam][i].size(); ++j) {
                            temp_pcl.at<double>(j, 0) = outputPoints_whole[cam][i][j].x;
                            temp_pcl.at<double>(j, 1) = outputPoints_whole[cam][i][j].y;
                            temp_pcl.at<double>(j, 2) = outputPoints_whole[cam][i][j].z;
                        }
                        cv::Mat newMat;
                        cv::Mat rowToAdd = cv::Mat::ones(1, temp_pcl.rows, temp_pcl.type());
                        cv::vconcat(temp_pcl.t(), rowToAdd, newMat);
                        pointcloud_vec[i] = newMat;
                        pointcloud_whole[cam][i] = newMat;
                    }
                }

                for (int i = 0; i < pointcloud_vec.size(); i++) {
                    if (cross_observation_matrix[i][cam] && !pointcloud_vec[i].empty()) {
                        //cv::Mat pointcloud_board = rototras_vec[cam][i].inv() * pointcloud_vec[i];
                        cv::Mat null_tras = (cv::Mat_<double>(3, 1) << 0, 0, 0);
                        cv::Mat svd;
                        getRotoTras<double>(svd_mat_vec[cam], null_tras, svd);
                        //cv::Mat pointcloud_world = svd*pointcloud_board;
                        //cv::Mat pointcloud_world = svd * pointcloud_whole[cam][i];
                        cv::Mat pointcloud_world = svd * pointcloud_vec[i];
                        // If read from triangulation
                        //pointcloud_world = pointcloud_vec[i];

                        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud <pcl::PointXYZ>);

                        // Fill the cloud with some points
                        for (int k = 0; k < pointcloud_world.cols; k++) {
                            cloud->points.push_back(
                                    pcl::PointXYZ(pointcloud_world.at<double>(0, k), pointcloud_world.at<double>(1, k),
                                                  pointcloud_world.at<double>(2, k)));
                            cloud->width = cloud->points.size();
                            cloud->height = 1;
                            cloud->is_dense = false;
                        }

                        // Create a PCLVisualizer object
                        /*pcl::visualization::PCLVisualizer viewer("3D Viewer");
                        viewer.setBackgroundColor(1, 1, 1);
                        viewer.addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
                        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3,
                                                                "sample cloud");
                        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloudColorHandler2(cloud, 0, 0, 0);
                        viewer.addPointCloud(cloud, cloudColorHandler2, "originalCloud");
                        viewer.addCoordinateSystem(1.0);
                        viewer.initCameraParameters();

                        // Main visualization loop
                        while (!viewer.wasStopped()) {
                            viewer.spinOnce(100);
                        }*/


                        // RANSAC
                        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
                        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
                        // Create the segmentation object
                        pcl::SACSegmentation <pcl::PointXYZ> seg;
                        // Optional
                        seg.setOptimizeCoefficients(true);
                        // Mandatory
                        seg.setModelType(pcl::SACMODEL_PLANE);
                        seg.setMethodType(pcl::SAC_RANSAC);


                        //seg.setDistanceThreshold (0.01);
                        seg.setDistanceThreshold(0.05);
                        seg.setInputCloud(cloud);
                        seg.segment(*inliers, *coefficients);

                        pcl::ExtractIndices <pcl::PointXYZ> extract;
                        pcl::PointCloud<pcl::PointXYZ>::Ptr inliersCloud(new pcl::PointCloud <pcl::PointXYZ>);
                        extract.setInputCloud(cloud);
                        extract.setIndices(inliers);
                        extract.filter(*inliersCloud);


                        if (inliers->indices.size() == 0) {
                            PCL_ERROR("Could not estimate a planar model for the given dataset.\n");
                        }
                        /*std::cerr << "Model coefficients: " << coefficients->values[0] << " "
                                  << coefficients->values[1] << " "
                                  << coefficients->values[2] << " "
                                  << coefficients->values[3] << std::endl;*/

                        cv::Mat normal_wrt_world = (cv::Mat_<double>(3, 1)
                                << coefficients->values[0], coefficients->values[1], coefficients->values[2]);
                        //cv::Mat normal_wrt_world = svd_mat*normal_wrt_board;
                        cv::Point3d normal_vec_wrt_world(normal_wrt_world.at<double>(0), normal_wrt_world.at<double>(1),
                                                         normal_wrt_world.at<double>(2));

                        cv::Point3d normal_world(0, 0, 1);
                        double scalar =
                                normal_world.x * normal_vec_wrt_world.x + normal_world.y * normal_vec_wrt_world.y +
                                normal_world.z * normal_vec_wrt_world.z;
                        /*pcl::visualization::PCLVisualizer::Ptr viewer2(new pcl::visualization::PCLVisualizer("3D Viewer PLANE"));
                        viewer2->setBackgroundColor(1, 1, 1);

                        // Add the original point cloud in white
                        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloudColorHandler(cloud, 0, 0, 0);
                        viewer2->addPointCloud(cloud, cloudColorHandler, "originalCloud");

                        // Add the inliers in red
                        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> inliersColorHandler(inliersCloud, 255, 0, 0);
                        viewer2->addPointCloud(inliersCloud, inliersColorHandler, "inliersCloud");
                        viewer2->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "inliersCloud");

                        // Add coordinate system and initialize camera view
                        viewer2->addCoordinateSystem(2.0);
                        viewer2->initCameraParameters();

                        // Main visualization loop
                        while (!viewer2->wasStopped()) {
                            viewer2->spinOnce(100);
                        }*/

                        d_board_vec[i] = coefficients->values[3];
                        ang_board_vec[i] = scalar;
                        //std::cout << "D board vec: " << d_board_vec[i] << std::endl;
                    }
                }

                for (int i = 0; i < number_of_waypoints; i++) {
                    if (cross_observation_matrix[i][cam] && !pointcloud_vec[i].empty()) {

                        cv::Mat null_tras = (cv::Mat_<double>(3, 1) << 0, 0, 0);
                        cv::Mat svd;
                        getRotoTras<double>(svd_mat_inv_vec[cam], null_tras, svd);
                        cv::Mat pointcloud_world = svd * rototras_vec[cam][i] * pointcloud_vec[i];
                        //cv::Mat pointcloud_world = pointcloud_vec[i];
                        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud <pcl::PointXYZ>);

                        // Fill the cloud with some points
                        for (int k = 0; k < pointcloud_world.cols; k++) {
                            double x = pointcloud_world.at<double>(0, k); // Access the first element
                            double y = pointcloud_world.at<double>(1, k); // Access the second element

                            double l2Norm = std::sqrt(x*x + y*y);
                            if(pointcloud_world.at<double>(2, k)< -0.20 && l2Norm<board_distance_whole[cam][i]) {
                                cloud->points.push_back(
                                        pcl::PointXYZ(pointcloud_world.at<double>(0, k),
                                                      pointcloud_world.at<double>(1, k),
                                                      pointcloud_world.at<double>(2, k)));
                                cloud->width = cloud->points.size();
                                cloud->height = 1;
                                cloud->is_dense = false;
                            }
                        }
                        if (cloud->points.size() < 4){
                            continue;
                        }
                        cv::Point3d normal_world(0, 0, 1);

                        // Create a PCLVisualizer object
                        /*pcl::visualization::PCLVisualizer viewer("3D Viewer");
                        viewer.setBackgroundColor(0, 0, 0);
                        viewer.addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
                        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10,
                                                                "sample cloud");
                        viewer.addCoordinateSystem(1.0);
                        viewer.initCameraParameters();*/

                        // RANSAC
                        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
                        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
                        // Create the segmentation object
                        pcl::SACSegmentation <pcl::PointXYZ> seg;
                        // Optional
                        seg.setOptimizeCoefficients(true);
                        // Mandatory
                        seg.setModelType(pcl::SACMODEL_PLANE);
                        seg.setMethodType(pcl::SAC_RANSAC);
                        seg.setDistanceThreshold(1-ground_plane_confidence);
                        seg.setInputCloud(cloud);
                        seg.segment(*inliers, *coefficients);

                        pcl::ExtractIndices <pcl::PointXYZ> extract;
                        pcl::PointCloud<pcl::PointXYZ>::Ptr inliersCloud(new pcl::PointCloud <pcl::PointXYZ>);
                        extract.setInputCloud(cloud);
                        extract.setIndices(inliers);
                        extract.filter(*inliersCloud);

                        if (inliers->indices.size() == 0) {
                            PCL_ERROR("Could not estimate a planar model for the given dataset.\n");
                        }



                        // Plane check
                        cv::Mat normal_wrt_world = (cv::Mat_<double>(3, 1)
                                << coefficients->values[0], coefficients->values[1], coefficients->values[2]);
                        cv::Point3d normal_vec_wrt_world(normal_wrt_world.at<double>(0), normal_wrt_world.at<double>(1),
                                                         normal_wrt_world.at<double>(2));
                        double scalar =
                                normal_world.x * normal_vec_wrt_world.x + normal_world.y * normal_vec_wrt_world.y +
                                normal_world.z * normal_vec_wrt_world.z;

                        d_camera_vec[i] = coefficients->values[3];
                        ang_camera_vec[i] = scalar;
                        //std::cout << "Scalar: " << scalar << std::endl;


                        std::ofstream csvFile(data_ + "/ground_plane_points" + std::to_string(cam+1)+ ".csv");

                        // Check if the file is open
                        if (csvFile.is_open()) {

                            // Iterate through the points in the inliers cloud
                            for (size_t index = 0; index < inliersCloud->points.size(); ++index) {

                                // Extract the original point
                                // Get the point
                                pcl::PointXYZ point = inliersCloud->points[index];

                                // Write the point's coordinates to the file, separated by commas
                                csvFile << point.x << "," << point.y << "," << point.z << "\n";
                            }

                            // Close the file
                            csvFile.close();
                        } else {
                            std::cerr << "Unable to open file for writing.\n";
                        }



                        /*std::cerr << "Model coefficients: " << coefficients->values[0] << " "
                                  << coefficients->values[1] << " "
                                  << coefficients->values[2] << " "
                                  << coefficients->values[3] << std::endl;*/

                        /*pcl::visualization::PCLVisualizer::Ptr viewer2(
                                new pcl::visualization::PCLVisualizer("3D Viewer PLANE"));
                        viewer2->setBackgroundColor(0, 0, 0);

                        // Add the original point cloud in white
                        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloudColorHandler(cloud, 255,
                                                                                                          255, 255);
                        viewer2->addPointCloud(cloud, cloudColorHandler, "originalCloud");

                        // Add the inliers in red
                        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> inliersColorHandler(
                                inliersCloud, 255, 0, 0);
                        viewer2->addPointCloud(inliersCloud, inliersColorHandler, "inliersCloud");
                        viewer2->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4,
                                                                  "inliersCloud");

                        // Add coordinate system and initialize camera view
                        viewer2->addCoordinateSystem(1.0);
                        viewer2->initCameraParameters();

                        // Main visualization loop
                        while (!viewer2->wasStopped()) {
                            viewer2->spinOnce(100);
                        }*/

                        /*pcl::visualization::PCLVisualizer viewer("Plane Finder");
                        viewer.setBackgroundColor(0, 0, 0);
                        viewer.addCoordinateSystem(1.0);
                        viewer.initCameraParameters();

                        int plane_id = 0;
                        bool planeFound = false; // Flag to track if any suitable plane was found
                        double bestScalar = -1; // Initialize to an impossible value to ensure any real scalar is better

                        pcl::ModelCoefficients::Ptr bestCoefficients(new pcl::ModelCoefficients);
                        pcl::PointCloud<pcl::PointXYZ>::Ptr remainingCloud = cloud; // Start with the full cloud

                        while (remainingCloud->points.size() > 0) {
                            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
                            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

                            pcl::SACSegmentation<pcl::PointXYZ> seg;
                            seg.setOptimizeCoefficients(true);
                            seg.setModelType(pcl::SACMODEL_PLANE);
                            seg.setMethodType(pcl::SAC_RANSAC);
                            seg.setDistanceThreshold(0.01);
                            seg.setInputCloud(remainingCloud);
                            seg.segment(*inliers, *coefficients);

                            if (inliers->indices.size() == 0) {
                                break; // No more planes can be found
                            }

                            // Visualize the plane
                            std::string planeLabel = "plane_" + std::to_string(plane_id);
                            //viewer.addPointCloud<pcl::PointXYZ>(remainingCloud, planeLabel);
                            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, planeLabel);

                            // Create a point for the plane normal visualization
                            pcl::PointXYZ plane_origin(
                                    0,0,0);
                            pcl::PointXYZ plane_normal_end(
                                    plane_origin.x + coefficients->values[0],
                                    plane_origin.y + coefficients->values[1],
                                    plane_origin.z + coefficients->values[2]);
                            std::string normalLabel = "normal_" + std::to_string(plane_id);
                            //viewer.addArrow(plane_normal_end, plane_origin, 1.0, 0, 0, false, normalLabel);


                            // Plane check
                            cv::Mat normal_wrt_world = (cv::Mat_<double>(3, 1)
                                    << coefficients->values[0], coefficients->values[1], coefficients->values[2]);
                            cv::Point3d normal_vec_wrt_world(normal_wrt_world.at<double>(0), normal_wrt_world.at<double>(1),
                                                             normal_wrt_world.at<double>(2));
                            double scalar = normal_world.x * normal_vec_wrt_world.x + normal_world.y * normal_vec_wrt_world.y +
                                            normal_world.z * normal_vec_wrt_world.z;
                            //std::cout << "Scalar: " << scalar << std::endl;

                            // Update best plane if this one is better
                            if (std::abs(scalar) > bestScalar) {
                                bestScalar = scalar;
                                *bestCoefficients = *coefficients;
                                planeFound = true;
                            }

                            plane_id++;

                            // Remove inliers from the cloud to search for the next plane
                            pcl::ExtractIndices<pcl::PointXYZ> extract;
                            extract.setInputCloud(remainingCloud);
                            extract.setIndices(inliers);
                            extract.setNegative(true); // Remove inliers
                            pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud(new pcl::PointCloud<pcl::PointXYZ>);
                            extract.filter(*tempCloud);
                            remainingCloud = tempCloud;


                            if (bestScalar > 0.995) { // Assuming you're looking for a plane almost parallel to Z-axis
                                std::cout << "Found a correct plane with confidence: " << bestScalar << std::endl;
                                std::cout << "Height: " << bestCoefficients->values[3] << std::endl;
                                pcl::PointXYZ arrow_origin(0.0, 0.0, 0.0);

                                // Arrow's end point is a scaled version of the plane's normal vector
                                // Scale factor can be adjusted as needed for visibility
                                pcl::PointXYZ arrow_end(
                                        bestCoefficients->values[0],
                                        bestCoefficients->values[1],
                                        bestCoefficients->values[2]);

                                // Add the arrow to the viewer with a different color, e.g., red
                                viewer.addArrow(arrow_end, arrow_origin, 0.0, 1.0, 0.0, false, "best_plane_normal");


                                break; // Stop if we find a plane that is almost vertical
                            }
                        }

                        //std::cout << "Best scalar: " << bestScalar << std::endl;

                        while (!viewer.wasStopped()) {
                            viewer.spinOnce(100);
                        }
                        viewer.removeAllPointClouds();
                        viewer.removeAllShapes();

                        if (planeFound) {
                            std::cerr << "Best model coefficients: " << bestCoefficients->values[0] << " "
                                      << bestCoefficients->values[1] << " "
                                      << bestCoefficients->values[2] << " "
                                      << bestCoefficients->values[3] << std::endl;
                            std::cout << "Best Scalar: " << bestScalar << std::endl;
                            d_camera_vec[i] = bestCoefficients->values[3];
                            ang_camera_vec[i] = bestScalar;
                        } else {
                            PCL_ERROR("Could not estimate a planar model close to vertical for the given dataset.\n");
                        }*/
                    }
                }
            }

            // Find the iterator to the highest value
            auto it_board = std::max_element(ang_board_vec.begin(), ang_board_vec.end());

            // Calculate the index
            int index_board = std::distance(ang_board_vec.begin(), it_board);
            double best_board_height = d_board_vec[index_board];

            // Find the iterator to the highest value
            auto it_cam = std::max_element(ang_camera_vec.begin(), ang_camera_vec.end());

            // Calculate the index
            int index_cam = std::distance(ang_camera_vec.begin(), it_cam);
            double best_cam_height = d_camera_vec[index_cam];

            double threshold = 0.995;

            for (int i = 0; i < number_of_waypoints; i++) {
                if (ang_board_vec[i] < threshold) {
                    d_board_vec[i] = 0;
                }
                if (ang_camera_vec[i] < threshold) {
                    d_camera_vec[i] = 0;
                }

            }

            d_camera_whole[cam] = d_camera_vec;
            d_board_whole[cam] = d_board_vec;
            ang_camera_whole[cam] = ang_camera_vec;
            ang_board_whole[cam] = ang_board_vec;
        }














        // ------------------------------------------ DATA AUGMENTATION ------------------------------------------
        /*for (int cam = 0; cam < number_of_cameras; cam++) {
            std::vector <cv::Mat> robot_perturbation;
            bool data_augmentation_function = false;
            bool data_perturbation_function = false;
            double std_dev = 0;
            double std_dev_t = 0.0;

            if (data_augmentation_function) {
                data_augmentation(poses_collected[cam], rototras_vec[cam], tvec_used[cam], rvec_used[cam],
                                  svd_mat_vec[cam],
                                  std_dev,
                                  std_dev_t, cross_observation_matrix);
            }

            if (data_perturbation_function) {
                double std_dev_tras = 0;
                double std_dev_rot = 0;
                data_perturbation(poses_collected[0], std_dev_tras, std_dev_rot);
            }
        }*/
    }


    // Get 3D checkerboard points
    std::vector <cv::Point3f> object_points;
    detector.getObjectPoints(object_points);

    // Set initial guess
    std::vector <cv::Mat> h2e_initial_guess_vec(number_of_cameras);
    cv::Mat b2ee_initial_guess;
    setInitialGuess(h2e_initial_guess_vec, b2ee_initial_guess, rototras_all, correct_poses, calib_info.getCalibSetup());

    // Print initial guess
    std::cout << "B2w: " << b2ee_initial_guess << std::endl;
    for (int i = 0; i < number_of_cameras; i++) {
        std::cout << "H2e: " << h2e_initial_guess_vec[i] << std::endl;
    }


    // ------------------------------------------ Calibration process ------------------------------------------
    std::vector <std::vector<cv::Mat>> multi_c2c_optimal_vec(number_of_cameras,
                                                             std::vector<cv::Mat>(number_of_cameras));
    std::vector <cv::Mat> multi_h2e_optimal_vec(number_of_cameras);
    cv::Mat multi_b2ee_optimal;
    std::vector <cv::Mat> multi_b2ee_optimal_vec;

    MultiHandEyeCalibrator multi_calibrator(number_of_waypoints, number_of_cameras, object_points, poses_collected[0],
                                            h2e_initial_guess_vec, b2ee_initial_guess, cross_observation_matrix,
                                            rvec_used, tvec_used, rototras_vec);
    //MobileHandEyeCalibrator mobile_calibrator(number_of_waypoints, number_of_cameras, object_points, poses_collected[0], h2e_initial_guess_vec, b2ee_initial_guess, cross_observation_matrix, rvec_used, tvec_used, rototras_vec, d_board_whole, d_camera_whole, relative_robot_poses[0], relative_cam_poses);

    std::vector <cv::Mat> optim_poses_collected(number_of_waypoints);
    std::vector <std::vector<cv::Mat>> optim_pnp_collected(number_of_cameras,
                                                           std::vector<cv::Mat>(number_of_waypoints));
    auto start_time_MULTI = std::chrono::high_resolution_clock::now();

    switch (calib_info.getCalibSetup()) {
        case 0:
            multi_calibrator.eyeInHandCalibration(camera_network_info, correct_corners, multi_h2e_optimal_vec,
                                                  multi_b2ee_optimal, multi_c2c_optimal_vec, images_collected);
            break;
        case 1:
            multi_calibrator.eyeOnBaseCalibration(camera_network_info, correct_corners, multi_h2e_optimal_vec,
                                                  multi_b2ee_optimal, multi_c2c_optimal_vec, selected_poses);

            break;
            /*case 2:
                mobile_calibrator.mobileCalibration(camera_network_info, correct_corners, multi_h2e_optimal_vec,
                                                    multi_b2ee_optimal, multi_c2c_optimal_vec,
                                                    optim_poses_collected, optim_pnp_collected);
                break;*/
    }

    auto end_time_MULTI = std::chrono::high_resolution_clock::now();
    auto duration_MULTI = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_MULTI - start_time_MULTI);



    /*for (int i = 0; i < number_of_cameras; i++) {
        //std::cout << "Multi h2e optimal: " << multi_h2e_optimal_vec[i].inv() << std::endl;
        cv::Mat temp1 = multi_h2e_optimal_vec[i];
        cv::Mat temp2 = multi_h2e_optimal_vec[i].inv();
        cv::Mat tras1, tras2, rot1, rot2;
        getTras<double>(temp1, tras1);
        getTras<double>(temp2, tras2);

        getRoto<double>(temp1, rot1);
        getRoto<double>(temp2, rot2);
        std::cout << "Tras1: " << tras1 << std::endl;
        std::cout << "Tras2: " << tras2 << std::endl;

        std::cout << "Rot1: " << rot1 << std::endl;
        std::cout << "Rot2: " << rot2 << std::endl;
    }*/

    auto start_time_mobile = std::chrono::high_resolution_clock::now();


    std::vector <std::vector<double>> translation_error_vector(number_of_cameras,
                                                               std::vector<double>(number_of_waypoints - 1, 1.0));
    double fixed_threshold = 0.001;
    double fixed_rotation = 0.5;
    int refinement_iteration = 0;

    if (calib_info.getCalibSetup() == 2){
        std::vector<cv::Mat> best_h2e_wor(number_of_cameras);
        std::vector<cv::Mat> best_h2e_wor_joint(number_of_cameras);
        MobileHandEyeCalibrator mobile_calibrator(number_of_waypoints, number_of_cameras, object_points,
                                                  poses_collected[0], h2e_initial_guess_vec, b2ee_initial_guess,
                                                  cross_observation_matrix, rvec_used, tvec_used, rototras_vec,
                                                  d_board_whole, d_camera_whole, relative_robot_poses,
                                                  relative_cam_poses);

        auto start_time_metroc = std::chrono::high_resolution_clock::now();
        mobile_calibrator.mobileCalibration(camera_network_info, correct_corners, multi_h2e_optimal_vec,
                                            multi_b2ee_optimal, multi_c2c_optimal_vec,
                                            optim_poses_collected, optim_pnp_collected);



        auto end_time_metroc = std::chrono::high_resolution_clock::now();
        auto duration_metroc = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_metroc - start_time_metroc);
        std::cout << "METROC CALIBRATION TIME: " << duration_metroc.count() << std::endl;

        for (int i = 0; i < number_of_cameras; i++){
            best_h2e_wor[i] = multi_h2e_optimal_vec[i];
        }


        mobile_calibrator.mobileJointCalibration(camera_network_info, correct_corners, multi_h2e_optimal_vec,
                                                 multi_b2ee_optimal, multi_c2c_optimal_vec,
                                                 optim_poses_collected, optim_pnp_collected);


        for (int i = 0; i < number_of_cameras; i++){
            best_h2e_wor_joint[i] = multi_h2e_optimal_vec[i];
        }

        std::vector<cv::Mat> best_h2e(number_of_cameras);
        auto start_time_ref = std::chrono::high_resolution_clock::now();
        while (hasNonZero(translation_error_vector, cross_observation_matrix)) {

            MobileHandEyeCalibrator mobile_calibrator_REF(number_of_waypoints, number_of_cameras, object_points,
                                                      poses_collected[0], h2e_initial_guess_vec, b2ee_initial_guess,
                                                      cross_observation_matrix, rvec_used, tvec_used, rototras_vec,
                                                      d_board_whole, d_camera_whole, relative_robot_poses,
                                                      relative_cam_poses);
            mobile_calibrator_REF.mobileCalibration(camera_network_info, correct_corners, multi_h2e_optimal_vec,
                                                multi_b2ee_optimal, multi_c2c_optimal_vec,
                                                optim_poses_collected, optim_pnp_collected);

            std::vector <cv::Mat> h2e_optimized(number_of_cameras);
            std::vector <cv::Mat> tvec_final_h2e(number_of_cameras);
            std::vector <cv::Mat> rvec_final_h2e(number_of_cameras);
            std::vector <std::vector<cv::Mat>> tvec_final_c2c(number_of_cameras,
                                                              std::vector<cv::Mat>(number_of_cameras));
            std::vector <std::vector<cv::Mat>> rvec_final_c2c(number_of_cameras,
                                                              std::vector<cv::Mat>(number_of_cameras));



            for (int i = 0; i < number_of_cameras; i++) {
                h2e_optimized[i] = multi_h2e_optimal_vec[i];
                //std::cout << "Multi h2e optimal: " << h2e_optimized[i] << std::endl;
                cv::Mat rotation;
                getTras<double>(h2e_optimized[i], tvec_final_h2e[i]);
                getRoto<double>(h2e_optimized[i], rotation);
                rvec_final_h2e[i] = rotationMatrixToEulerAngles<double>(rotation);
                //std::cout << "Tras: " << tvec_final_h2e[i] << std::endl;
                //std::cout << "Rot: " << rvec_final_h2e[i] << std::endl;
            }

            // RANSAC for refinement


            for (int i = 0; i < number_of_cameras; i++) {
                double tras_mean = 0.0;
                double rot_mean = 0.0;
                int counter = 0;
                std::vector <cv::Mat> robot_chain;
                std::vector <cv::Mat> pnp_chain;
                std::cout << "########## Camera " << std::to_string(i + 1) << " ############" << std::endl;
                for (int j = 0; j < number_of_waypoints - 1; j++) {
                    if (cross_observation_matrix[j][i] && cross_observation_matrix[j + 1][i]) {
                        cv::Mat chain1 = relative_robot_poses[i][j] * h2e_optimized[i];
                        cv::Mat chain2 = h2e_optimized[i] * relative_cam_poses[i][j];
                        double translation_error, rotation_error;
                        translationError(chain1, chain2, translation_error);
                        rotationError(chain1, chain2, rotation_error);
                        //std::cout << "########### Error from " << std::to_string(j) << " to " << std::to_string(j + 1)
                        //          << " ##########" << std::endl;
                        //std::cout << "Translation error: " << translation_error << std::endl;
                        double tx = abs(chain1.at<double>(0, 3) - chain2.at<double>(0, 3));
                        double ty = abs(chain1.at<double>(1, 3) - chain2.at<double>(1, 3));
                        double tz = abs(chain1.at<double>(2, 3) - chain2.at<double>(2, 3));
                        //std::cout << "tx: " << tx << std::endl;
                        //std::cout << "ty: " << ty << std::endl;
                        //std::cout << "tz: " << tz << std::endl;
                        //std::cout << "Rotation error: " << rotation_error << std::endl;

                        if (translation_error > fixed_threshold || rotation_error > fixed_rotation || ty>tx) {
                            std::cout << "------------- Camera " << std::to_string(i + 1) << " wp: "
                            << std::to_string(j) << " trans error: " << translation_error
                            << " rot error: " << rotation_error << std::endl;
                            translation_error_vector[i][j] = translation_error;
                            cross_observation_matrix[j][i] = 0;
                        } else {
                            translation_error_vector[i][j] = 0.0;
                        }
                        robot_chain.push_back(
                                (cv::Mat_<double>(2, 1) << chain1.at<double>(0, 3), chain1.at<double>(1, 3)));
                        pnp_chain.push_back(
                                (cv::Mat_<double>(2, 1) << chain2.at<double>(0, 3), chain2.at<double>(1, 3)));
                    } else {
                        translation_error_vector[i][j] = 0.0;
                    }
                }
            }



            for (int i = 0; i < number_of_cameras; i++) {
                for (int j = 0; j < number_of_cameras; j++) {
                    cv::Mat rotation;
                    getTras<double>(h2e_optimized[i].inv() * h2e_optimized[j], tvec_final_c2c[i][j]);
                    getRoto<double>(h2e_optimized[i].inv() * h2e_optimized[j], rotation);
                    rvec_final_c2c[i][j] = rotationMatrixToEulerAngles<double>(rotation);
                    std::cout << "Tras from " << std::to_string(j + 1) << " to " << std::to_string(i + 1) << ": "
                              << tvec_final_c2c[i][j] << std::endl;
                }
            }


            std::string filename = "/home/davide/davide_ws/src/robot_mounted_checkerboard/METRIC_Calibrator/data/real_mobile_robot/METROC/results.txt";
            std::string filename_csv = "/home/davide/davide_ws/src/robot_mounted_checkerboard/METRIC_Calibrator/data/real_mobile_robot/METROC/results.csv";

            std::vector <Transformation> h2e(number_of_cameras);
            for (int i = 0; i < number_of_cameras; i++) {
                h2e[i] = {tvec_final_h2e[i].at<double>(0), tvec_final_h2e[i].at<double>(1),
                          tvec_final_h2e[i].at<double>(2), rvec_final_h2e[i].at<double>(0),
                          rvec_final_h2e[i].at<double>(1), rvec_final_h2e[i].at<double>(2)};
            }
            std::vector <std::vector<Transformation>> cam2cam(number_of_cameras,
                                                              std::vector<Transformation>(number_of_cameras));
            for (int i = 0; i < number_of_cameras; i++) {
                for (int j = 0; j < number_of_cameras; j++) {
                    cam2cam[i][j] = {tvec_final_c2c[i][j].at<double>(0), tvec_final_c2c[i][j].at<double>(1),
                                     tvec_final_c2c[i][j].at<double>(2), rvec_final_c2c[i][j].at<double>(0),
                                     rvec_final_c2c[i][j].at<double>(1), rvec_final_c2c[i][j].at<double>(2)};
                }
            }

            saveTransformations(filename, dataset, h2e, cam2cam);
            appendTransformationsToFile(filename_csv, dataset, start_index, end_index, h2e, cam2cam);


            //fixed_threshold = fixed_threshold-0.10*fixed_threshold;
            std::cout << "THRESHOLD: " << fixed_threshold << std::endl;
            std::cout << "REFINEMENT ITERATION: " << refinement_iteration << std::endl;
            refinement_iteration++;
            //fixed_threshold = fixed_threshold - 0.2 * fixed_threshold;
            //fixed_rotation = fixed_rotation - 0.2 * fixed_rotation;

            for (int i = 0; i< number_of_cameras; i++){
                best_h2e[i] = multi_h2e_optimal_vec[i];
            }
        }

        auto end_time_ref = std::chrono::high_resolution_clock::now();
        auto duration_ref = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_ref - start_time_ref);
        std::cout << "REF CALIBRATION TIME: " << duration_ref.count() << std::endl;



        int max_iteration = 1000;
        int minInliers = 30;        // At each run we randomly select 10 observations
        double threshold = 0.05;

        // Count how many observations are available for each cam
        std::vector<int> counter_obs_vec(number_of_cameras);
        for (int i = 0; i < number_of_cameras; i++) {
            std::cout << "Camera " << std::to_string(i+1) << std::endl;
            int counter_obs = 0;
            for (int j = 0; j < number_of_waypoints-1; j++){
                if (!relative_cam_poses[i][j].empty()){
                    counter_obs++;
                }
            }
            counter_obs_vec[i] = counter_obs;
        }

        std::vector<int> good_number(number_of_cameras);
        for (int i = 0; i < number_of_cameras; i++){
            good_number[i] = static_cast<int>(0.8*counter_obs_vec[i]); // A good number of inliers is the 80% of total images
        }

        std::vector<double> best_error = std::vector<double>(number_of_cameras, 10);

        std::vector<cv::Mat> best_X(number_of_cameras);

        // If I want at each run a new random subset

        /*for (int iter = 0; iter < max_iteration; iter++){
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

            std::default_random_engine generator(seed);
            std::uniform_int_distribution<size_t> distribution(0, number_of_waypoints - 2);
            std::vector<std::vector<cv::Mat>> selected_A(number_of_cameras, std::vector<cv::Mat> (number_of_waypoints-1));
            std::vector<std::vector<cv::Mat>> selected_B(number_of_cameras, std::vector<cv::Mat> (number_of_waypoints-1));
            std::vector <std::vector<int>> cross_observation_matrix_RANSAC(number_of_waypoints, std::vector<int>(number_of_cameras,0));


            // Select a random subset of A-B pairs
            selectRandomSubsetAXXB_RANSAC(selected_A, selected_B, cross_observation_matrix_RANSAC, minInliers, number_of_cameras, generator, distribution, relative_robot_poses, relative_cam_poses);

            for (int i = 0; i < number_of_cameras; i++){
                int counter = 0;
                for (int j = 0; j< selected_A[i].size(); j++){
                    if (!selected_A[i][j].empty()){
                        counter++;
                    }
                }
                std::cout << "Inliers: " << counter << std::endl;
            }




            std::vector<int> inliers(number_of_cameras, 0);
            MobileHandEyeCalibrator mobile_calibrator_init(number_of_waypoints-1, number_of_cameras, object_points,
                                                      poses_collected[0], h2e_initial_guess_vec, b2ee_initial_guess,
                                                      cross_observation_matrix_RANSAC, rvec_used, tvec_used, rototras_vec,
                                                      d_board_whole, d_camera_whole, selected_A,
                                                      selected_B);
            mobile_calibrator_init.mobileCalibration(camera_network_info, correct_corners, multi_h2e_optimal_vec,
                                                multi_b2ee_optimal, multi_c2c_optimal_vec,
                                                optim_poses_collected, optim_pnp_collected);

            std::vector <cv::Mat> h2e_optimized(number_of_cameras);
            std::vector <cv::Mat> tvec_final_h2e(number_of_cameras);
            std::vector <cv::Mat> rvec_final_h2e(number_of_cameras);
            std::vector <std::vector<cv::Mat>> tvec_final_c2c(number_of_cameras,
                                                              std::vector<cv::Mat>(number_of_cameras));
            std::vector <std::vector<cv::Mat>> rvec_final_c2c(number_of_cameras,
                                                              std::vector<cv::Mat>(number_of_cameras));


            std::cout << "############## RANSAC ITERATION " << std::to_string(iter) << " ##############" << std::endl;
            for (int i = 0; i < number_of_cameras; i++) {
                h2e_optimized[i] = multi_h2e_optimal_vec[i];
                //std::cout << "Multi h2e initial: " << h2e_optimized[i] << std::endl;
            }

            for (int i = 0; i < number_of_cameras; i++) {
                //std::cout << "ERROR ANALYSIS: " << std::to_string(i+1) << std::endl;
                for (int j = 0; j < number_of_waypoints - 1; j++) {
                    if (!relative_cam_poses[i][j].empty()) {
                        double tras_error = computeTransError_AXXB(relative_robot_poses[i][j], relative_cam_poses[i][j],
                                                                   multi_h2e_optimal_vec[i]);
                        if (tras_error < threshold) {
                            //std::cout << "Added wp " << std::to_string(j) << ": " << tras_error << std::endl;
                            inliers[i]++;
                            selected_A[i][j] = relative_robot_poses[i][j];
                            selected_B[i][j] = relative_cam_poses[i][j];
                            cross_observation_matrix_RANSAC[j][i] = 1;
                            cross_observation_matrix_RANSAC[j + 1][i] = 1;
                        }
                    }
                }
            }

            for (int i = 0; i < number_of_cameras; i++){
                std::cout << "###### Camera" << std::to_string(i+1) << " #########" << std::endl;
                int counter = 0;
                for (int j = 0; j< selected_A[i].size(); j++){
                    if (!selected_A[i][j].empty()){
                        //std::cout << "A: " << selected_A[i][j] << std::endl;
                        counter++;
                    }
                }
                std::vector<double> this_error = computeTransAverageError_AXXB(selected_A, selected_B, multi_h2e_optimal_vec);
                //std::cout << "Camera " << std::to_string(i+1) << ": " << this_error[i] << std::endl;
                //std::cout << "Inliers: " << inliers[i] << std::endl;
            }

            bool inliers_sufficiently_large = false;
            for (int i = 0; i < number_of_cameras; i++){
                if (inliers[i]>good_number[i]){
                    inliers_sufficiently_large = true;
                }
            }

            if (inliers_sufficiently_large) {
                std::cout << "##### Inliers sufficiently large #####" << std::endl;
                MobileHandEyeCalibrator mobile_calibrator_ransac(selected_A[0].size(), number_of_cameras, object_points,
                                                                 poses_collected[0], h2e_initial_guess_vec,
                                                                 b2ee_initial_guess,
                                                                 cross_observation_matrix_RANSAC, rvec_used, tvec_used,
                                                                 rototras_vec,
                                                                 d_board_whole, d_camera_whole, selected_A,
                                                                 selected_B);
                mobile_calibrator_ransac.mobileCalibration(camera_network_info, correct_corners, multi_h2e_optimal_vec,
                                                           multi_b2ee_optimal, multi_c2c_optimal_vec,
                                                           optim_poses_collected, optim_pnp_collected);

                std::vector<double> this_error = computeTransAverageError_AXXB(selected_A, selected_B, multi_h2e_optimal_vec);
                for (int i = 0; i < number_of_cameras; i++){
                    //std::cout << "Tras camera " << std::to_string(i+1) << ": " << this_error[i] << std::endl;
                    if (this_error[i]<best_error[i]){
                        best_X[i] = multi_h2e_optimal_vec[i];
                        //std::cout << "BEST X: " << best_X[i] << std::endl;
                        best_error[i] = this_error[i];
                    }
                }
            }

        }*/
        std::cout << "#################################" << std::endl;
        for (int i = 0; i < number_of_cameras; i++) {
            std::cout << "Best h2e optimal RANSAC: " << best_X[i] << std::endl;
        }
        std::cout << "#################################" << std::endl;
        for (int i = 0; i < number_of_cameras; i++) {
            std::cout << "Best h2e optimal REFINEMENT: " << best_h2e[i] << std::endl;
        }
        std::vector<cv::Mat> cam2cam_ref;
        for (int i = 0; i < number_of_cameras; i++){
            for (int j = 0; j < number_of_cameras; j++){
                if (i!=j) {
                    cam2cam_ref.push_back(best_h2e[i].inv() * best_h2e[j]);
                }
            }
        }

        saveTranslationVectors(best_h2e, cam2cam_ref, data_ + "/refinement_.txt");



        cv::Mat tras1 = (cv::Mat_<double>(1,3) << 0.492, 0.018, 0.450);
        cv::Mat rot1 = (cv::Mat_<double>(1,3) << -1.571, -0.000, -1.571);
        cv::Mat tras2 = (cv::Mat_<double>(1,3) << 0.536, -0.439, 0.501);
        cv::Mat rot2 = (cv::Mat_<double>(1,3) << -1.571, -0.000, -1.571);
        cv::Mat tras3 = (cv::Mat_<double>(1,3) << 0.511, 0.367, 0.495);
        cv::Mat rot3 = (cv::Mat_<double>(1,3) << -1.571, -0.000, -1.571);
        cv::Mat G1,G2,G3;
        cv::Mat rot_mat1 = eulerAnglesToRotationMatrix<double>(rot1);
        cv::Mat rot_mat2 = eulerAnglesToRotationMatrix<double>(rot2);
        cv::Mat rot_mat3 = eulerAnglesToRotationMatrix<double>(rot3);
        getRotoTras<double>(rot_mat1, tras1, G1);
        getRotoTras<double>(rot_mat2, tras2, G2);
        getRotoTras<double>(rot_mat3, tras3, G3);
        std::vector<cv::Mat> gt_vec(number_of_cameras), gt_rot_vec(number_of_cameras), gt_tras_vec(number_of_cameras);
        gt_vec[0] = G1;
        gt_vec[1] = G2;
        gt_vec[2] = G3;
        gt_tras_vec[0] = tras1;
        gt_tras_vec[1] = tras2;
        gt_tras_vec[2] = tras3;
        gt_rot_vec[0] = rot1;
        gt_rot_vec[1] = rot2;
        gt_rot_vec[2] = rot3;

        std::vector<cv::Mat> cam2cam_gt;
        for (int i = 0; i < number_of_cameras; i++){
            for (int j = 0; j < number_of_cameras; j++){
                if (i!=j) {
                    cam2cam_gt.push_back(gt_vec[i].inv() * gt_vec[j]);
                }
            }
        }

        /*for (int i = 0; i < number_of_cameras; i++){
            double tras_error_h2e, rot_error_h2e;
            cv::Mat translation_est, rotation_est;
            getTras<double>(best_h2e_wor[i], translation_est);
            getRoto<double>(best_h2e_wor[i], rotation_est);
            cv::Mat euler_rot = rotationMatrixToEulerAngles<double>(rotation_est);
            tras_error_h2e = trasError(gt_tras_vec[i], translation_est);
            rot_error_h2e = trasError(gt_rot_vec[i], euler_rot);
            std::cout << "Tras h2e error: " << tras_error_h2e << std::endl;
            std::cout << "Rot h2e error: " << rot_error_h2e << std::endl;
            for (int j = 0; j< number_of_cameras; j++){
                cv::Mat cam2cam = best_h2e_wor[i].inv()*best_h2e_wor[j];
                cv::Mat trans_c2c, rot_c2c;
                getTras<double>(cam2cam, trans_c2c);
                getRoto<double>(cam2cam, rot_c2c);
                cv::Mat euler_c2c = rotationMatrixToEulerAngles<double>(rot_c2c);
                double tras_error_c2c = trasError(gt_tras_vec[i], translation_est);
                double rot_error_c2c = trasError(gt_rot_vec[i], euler_c2c);
                std::cout << "Tras c2c error: " << tras_error_c2c << std::endl;
                std::cout << "Rot c2c error: " << rot_error_c2c << std::endl;

            }
        }*/

        saveTranslationVectors(gt_vec, cam2cam_gt, data_ + "/gt_.txt");


        std::cout << "#################################" << std::endl;
        for (int i = 0; i < number_of_cameras; i++) {
            std::cout << "Best h2e optimal : " << best_h2e_wor[i] << std::endl;
            cv::Mat translation_est, rotation_est;
            getTras<double>(best_h2e_wor[i], translation_est);
            getRoto<double>(best_h2e_wor[i], rotation_est);
            cv::Mat euler_rot = rotationMatrixToEulerAngles<double>(rotation_est);
            //std::cout << "ESTIMATED TRAS: " << translation_est << std::endl;
            //std::cout << "ESTIMATED EULER: " << euler_rot << std::endl;
            bool sensorx2car = false;
            if (sensorx2car) {
                cv::Mat rotation;
                cv::Mat temp = (cv::Mat_<double>(4, 4) << 0, 0, 1, 0,
                        -1, 0, 0, 0,
                        0, -1, 0, 0);
                getRoto<double>(temp * best_h2e_wor[i].inv(), rotation);
                cv::Mat euler = rotationMatrixToEulerAngles<double>(rotation);
                cv::Mat euler_inv = rotationMatrixToEulerAngles<double>(rotation.inv());
                std::cout << "Euler rad: " << euler_inv << std::endl;
                euler.at<double>(0) = 180 * euler.at<double>(0) / 3.14;
                euler.at<double>(1) = 180 * euler.at<double>(1) / 3.14;
                euler.at<double>(2) = 180 * euler.at<double>(2) / 3.14;
                std::cout << "Euler deg: " << euler << std::endl;
            }

            //CAMERA2CAMERA
            for (int j = 0; j < number_of_cameras; j++){
                cv::Mat cam2cam = best_h2e_wor[i].inv()*best_h2e_wor[j];
                cv::Mat trans_c2c, rot_c2c;
                getTras<double>(cam2cam, trans_c2c);
                getRoto<double>(cam2cam, rot_c2c);
                cv::Mat euler_c2c = rotationMatrixToEulerAngles<double>(rot_c2c);
                //std::cout << "######## Cam " << std::to_string(j+1) << " to cam " << std::to_string(i+1) << std::endl;
                //std::cout << "Tras: " << trans_c2c << std::endl;
                //std::cout << "Rot: " << euler_c2c << std::endl;
            }
        }

        std::vector<cv::Mat> cam2cam_no_ref;
        for (int i = 0; i < number_of_cameras; i++){
            for (int j = 0; j < number_of_cameras; j++){
                if (i!=j) {
                    cam2cam_no_ref.push_back(best_h2e_wor[i].inv() * best_h2e_wor[j]);
                }
            }
        }
        std::vector<cv::Mat> cam2cam_no_ref_joint;
        for (int i = 0; i < number_of_cameras; i++){
            for (int j = 0; j < number_of_cameras; j++){
                if (i!=j) {
                    cam2cam_no_ref_joint.push_back(best_h2e_wor_joint[i].inv() * best_h2e_wor_joint[j]);
                }
            }
        }

        saveTranslationVectors(best_h2e_wor_joint, cam2cam_no_ref_joint, data_ + "/jointmetroc_.txt");
        saveTranslationVectors(best_h2e_wor, cam2cam_no_ref, data_ + "/metroc_.txt");

    }


    auto end_time_mobile = std::chrono::high_resolution_clock::now();
    auto duration_mobile = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_mobile - start_time_mobile);
    std::cout << "Mobile multi cam optimization: " << duration_mobile.count() << " milliseconds" << std::endl;

    bool opencv = false;
    auto start_time_opencv = std::chrono::high_resolution_clock::now();
    if (opencv){

        std::vector<cv::Mat> h2e_opencv(number_of_cameras);
        std::fill(h2e_opencv.begin(), h2e_opencv.end(), cv::Mat());
        std::vector<cv::Mat> cam2cam_opencv;
        for (int i = 0; i < number_of_cameras; i++){

            for (int j = 0; j < number_of_cameras; j++){
                std::vector<std::vector<cv::Point2f>> imagePoints_ref;
                std::vector<std::vector<cv::Point2f>> imagePoints;
                std::vector<std::vector<cv::Point3f>> objectPoints_vec;
                cv::Size image_size;
                if (i!=j) {
                    for (int k = 0; k < number_of_waypoints; k++) {
                        if (cross_observation_matrix[k][i] && cross_observation_matrix[k][j]) {
                            objectPoints_vec.push_back(object_points);
                            imagePoints_ref.push_back(correct_corners[k][i]);
                            imagePoints.push_back(correct_corners[k][j]);
                            image_size = images_collected[i][k].size();
                        }
                    }


                    cv::Mat R, T, E, F;
                    cv::Mat K1 = camera_network_info[i].getCameraMatrix();
                    cv::Mat K2 = camera_network_info[j].getCameraMatrix();
                    cv::Mat d1 = camera_network_info[i].getDistCoeff();
                    cv::Mat d2 = camera_network_info[j].getDistCoeff();
                    double rms = cv::stereoCalibrate(objectPoints_vec, imagePoints_ref, imagePoints, K1,
                                                     d1, K2, d2,
                                                     image_size, R, T, E, F, cv::CALIB_FIX_ASPECT_RATIO +
                                                                             cv::CALIB_ZERO_TANGENT_DIST +
                                                                             cv::CALIB_USE_INTRINSIC_GUESS +
                                                                             cv::CALIB_SAME_FOCAL_LENGTH +
                                                                             cv::CALIB_RATIONAL_MODEL,
                                                     cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 100, 1e-5));

                    std::cout << "######### Cam " << std::to_string(j+1) << "  to cam " << std::to_string(i+1) << std::endl;
                    std::cout << "R: " << R << std::endl;
                    cv::Mat euler_cv = rotationMatrixToEulerAngles<double>(R);
                    std::cout << "t: " << T << std::endl;
                    cv::Mat opencv_rototras;
                    getRotoTras<double>(R, T, opencv_rototras);
                    cam2cam_opencv.push_back(opencv_rototras.inv());
                }
            }
        }
        saveTranslationVectors(h2e_opencv, cam2cam_opencv, data_ + "/opencv_.txt");
    }
    auto end_time_opencv = std::chrono::high_resolution_clock::now();
    auto duration_opencv = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_opencv - start_time_opencv);
    std::cout << "OPENCV CALIBRATION TIME: " << duration_opencv.count() << std::endl;





    // Visualize error:
    //metric_AX_XB(h2e_optimized, poses_collected, rototras_vec, cross_observation_matrix);
    /*for (int i = 0; i < number_of_cameras; i++) {
        std::cout << "Multi h2e optimal: " << multi_h2e_optimal_vec[i].inv() << std::endl;
    }
    std::cout << "Multi b2ee optimal: " << multi_b2ee_optimal << std::endl;
*/


    multi_b2ee_optimal_vec.push_back(multi_b2ee_optimal);
    cv::Mat multi_b2ee_optimal_TMP = multi_b2ee_optimal;
    std::string t_b2e = type2str(multi_b2ee_optimal_TMP.type());


    std::vector<double> vec_trans_noisy, vec_rot_noisy, vec_trans_opt, vec_rot_opt;

    for (int i = 0; i < number_of_waypoints; i++){
        //optim_poses_collected[i].convertTo(optim_poses_collected[i], CV_64F);
        poses_collected[0][i].convertTo(poses_collected[0][i], CV_64F);
    }

    /*int size = robot_perturbation.size();
    for (int i = 0; i < number_of_waypoints - size; i++){
        cv::Mat identityMatrix = cv::Mat::eye(4, 4, CV_64F);
        robot_perturbation.insert(robot_perturbation.begin(), identityMatrix);
    }*/

    /*for (int i = 0; i < number_of_cameras; i++){
        double translation_cam = 0;
        double rotation_cam = 0;
        for(int j = 0; j < number_of_waypoints; j++){
            if(cross_observation_matrix[j][i]) {
                std::cout << "Iteration " << std::to_string(j) << std::endl;
                // A*X=Z*B
                //cv::Mat ax = optim_poses_collected[j].inv() * h2e_optimized[i];
                //cv::Mat zb = multi_b2ee_optimal * optim_pnp_collected[i][j];
                cv::Mat ax = optim_poses_collected[j] * h2e_gt1;
                cv::Mat zb = b2e_gt * optim_pnp_collected[i][j];  // T^{W}_{B}*T^{B}_{C}
                std::cout << "AX: " << ax << std::endl;
                std::cout << "ZB: " << zb << std::endl;

                double translation, rotation;
                translationError(ax, zb, translation);
                rotationError(ax, zb, rotation);
                std::cout << "Translation error: " << translation << std::endl;
                translation_cam += translation;
                std::cout << "Rotation error: " << rotation << std::endl;
                rotation_cam += rotation;
                // B' = B(X^-1*Q*X)
                //std::cout << "B': " << optim_pnp_collected[i][j] << std::endl;
                //std::cout << "B(X^-1*Q*X): " << rototras_vec[i][j].inv()*(h2e_optimized[i].inv()*robot_perturbation[j]*h2e_optimized[i]) << std::endl;
            }
        }
        std::cout << "Average translation error: " << translation_cam/number_of_waypoints << std::endl;
        std::cout << "Average rotation error: " << rotation_cam/number_of_waypoints << std::endl;
    }*/


    for (int i = 0; i < original_poses[0].size(); i++){
        original_poses[0][i].convertTo(original_poses[0][i], CV_64F);
    }


    //std::cout << "Parallel single cam optimization: " << duration_SINGLE_tot << " milliseconds" << std::endl;
    std::cout << "Multi cam optimization: " << duration_MULTI.count() << " milliseconds" << std::endl;

    // Get GT if available
    std::vector<cv::Mat> gt_vec(number_of_cameras);
    std::vector<std::vector<cv::Mat>> gt_c2c_vec(number_of_cameras, std::vector<cv::Mat>(number_of_cameras));

    if (calib_info.getGt())
        readGt(number_of_cameras, data_, gt_vec, gt_c2c_vec);


    // Multi camera calibration results
    std::string calibration_method = "multi";
    createFolder(data_ +  "/results_" + calibration_method);

    // Try to open the txt file for the comparison with the ground truth
    std::ofstream output_file_multi;
    if (calib_info.getGt()) {
        output_file_multi.open(data_ + "/results_" + calibration_method + "/gt_error.txt");
        if (!output_file_multi.is_open()) {
            std::cerr << "Unable to open the file." << std::endl;
        }
    }


    // Save the optimal b2ee to the corresponding file
    mat2Csv(data_ + "/results_" + calibration_method + "/estimated_b2ee_mat.csv", multi_b2ee_optimal.inv());
    // Initialize the metric object for each camera
    for (int i = 0; i < number_of_cameras; i++) {
        std::cout << "Camera to robot: " << multi_h2e_optimal_vec[i].inv() << std::endl;
        //std::cout << "Camera to robot inv: " << multi_h2e_optimal_vec[i] << std::endl;
        
        Eigen::Vector3d translation;
        Eigen::Quaterniond quaternion;
        if (calibration_setup == 0){
            multi_h2e_optimal_vec[i] = multi_h2e_optimal_vec[i].inv();
        }
        std::cout << "Hand with respect to the camera" << std::endl;
        extractTranslationAndRotation(multi_h2e_optimal_vec[i].inv(), translation, quaternion);
        std::cout << "Camera with respect to the hand" << std::endl;
        extractTranslationAndRotation(multi_h2e_optimal_vec[i], translation, quaternion);
        //std::cout << "Board to hand: " << multi_b2ee_optimal_vec[0] << std::endl;
        //multi_h2e_optimal_vec[i] = multi_h2e_optimal_vec[i].inv();
    }

    
    Metrics metric(object_points, multi_h2e_optimal_vec, multi_b2ee_optimal_vec, camera_network_info, calib_info, data_, calibration_method, cross_observation_matrix);

    // Reproject corners
    number_of_waypoints = images_collected[0].size();
    std::vector<std::vector<std::vector<cv::Point2f>>> corner_points_reprojected(number_of_cameras, std::vector<std::vector<cv::Point2f>>(number_of_waypoints));
    metric.projectCorners(correct_corners, original_poses[0], images_collected, corner_points_reprojected);
    //metric.projectCorners(correct_corners, original_poses[0], images_collected, corner_points_reprojected);

    // Save reprojection error
    metric.reprojectionError(correct_corners, corner_points_reprojected);
    double trans_multi_err_single = 0.0;
    double rot_multi_err_single = 0.0;
    for (int i = 0; i < number_of_cameras; i++) {
        // Save the optimal h2e to the corresponding file
        std::cout << "Camera: " << i+1 << std::endl;
        mat2Csv(data_ + "/results_" + calibration_method + "/estimated_h2e_mat_" + std::to_string(i+1) + ".csv", multi_h2e_optimal_vec[i].inv());
        if (calib_info.getGt()) {
            output_file_multi << "World to camera " << std::to_string(i+1) << std::endl;
            std::cout << "GT VEC: " << gt_vec[i] << std::endl;
            std::vector<double> cam2world_error = errorMetric(gt_vec[i], multi_h2e_optimal_vec[i].inv());
            output_file_multi << "Translation error: " << cam2world_error[0] << std::endl;
            output_file_multi << "Rotation error: " << cam2world_error[1] << std::endl;
            trans_multi_err_single += cam2world_error[0];
            rot_multi_err_single += cam2world_error[1];
            /*for (int j = 0; j < number_of_cameras; j++) {
                if (i != j) {
                    output_file_multi << "Camera " << std::to_string(j+1) << " to camera" << std::to_string(i+1) << std::endl;
                    std::vector<double> cam2cam_error = errorMetric(gt_c2c_vec[i][j], multi_h2e_optimal_vec[i] * multi_h2e_optimal_vec[j].inv());
                }
            }*/
        }
    }

    trans_multi_err_single = trans_multi_err_single/number_of_cameras;
    rot_multi_err_single = rot_multi_err_single/number_of_cameras;

    output_file_multi << "Average translation error: " << trans_multi_err_single << std::endl;
    output_file_multi << "Average rotation error: " << rot_multi_err_single << std::endl;

    output_file_multi.close();

    std::ofstream output_metric_multi;

    /*if (calib_info.getMetric()) {
        output_metric_multi.open(data_ + "/results_" + calibration_method + "/metric_error.txt");
        if (!output_metric_multi.is_open()) {
            std::cerr << "Unable to open the file." << std::endl;
        }
        double translation_error_tot = 0.0;
        double rotation_error_tot = 0.0;
        for (int i = 0; i < number_of_cameras; i++) {
            double translation_error_cam = 0.0;
            double rotation_error_cam = 0.0;
            int counter = 0;
            for (int j = 0; j < number_of_waypoints; j++) {
                if (cross_observation_matrix[j][i]){
                    cv::Mat rot_pnp;
                    rototras_vec[i][j].convertTo(rot_pnp, CV_64F);
                    cv::Mat robot_chain = poses_collected[i][j]*multi_b2ee_optimal;
                    cv::Mat camera_chain = multi_h2e_optimal_vec[i].inv()*rot_pnp;
                    double translation_error, rotation_error;
                    translationError(robot_chain, camera_chain, translation_error);
                    rotationError(robot_chain, camera_chain, rotation_error);
                    translation_error_cam += translation_error;
                    rotation_error_cam += rotation_error;
                    counter ++;
                }
            }
            translation_error_cam = translation_error_cam/counter;
            rotation_error_cam = rotation_error_cam/counter;
            translation_error_tot += translation_error_cam;
            rotation_error_tot += rotation_error_cam;
        }

        translation_error_tot = translation_error_tot/number_of_cameras;
        rotation_error_tot = rotation_error_tot/number_of_cameras;

        output_metric_multi << "Translation error AX=ZB: " << translation_error_tot*1000 << " mm" << std::endl;
        output_metric_multi << "Rotation error AX=ZB: " << rotation_error_tot << " deg" << std::endl;

        output_metric_multi.close();
    }*/




    /*for (int i = 0; i<poses_collected[0].size();i++) {
        if (cross_observation_matrix[i][0]) {

            // NO NOISE
            std::cout << "############ TEST AX=ZB ############ " << std::endl;
            cv::Mat rob_pose = poses_collected[0][i];
            cv::Mat pnp_pose = rototras_vec[0][i].inv();
            std::cout << "A: " << rob_pose << std::endl;
            std::cout << "B: " << pnp_pose << std::endl;
            cv::Mat ax = rob_pose * gt1;
            cv::Mat zb = b2e_gt * pnp_pose;

            cv::Mat rot_poses, rot_pnp, tras_poses, tras_pnp;
            getRoto<double>(rob_pose, rot_poses);
            getRoto<double>(pnp_pose, rot_pnp);
            std::cout << "Rotation A: " << rot_poses << std::endl;
            std::cout << "Rotation B: " << rot_pnp << std::endl;

            cv::Mat euler_angle_A = rotationMatrixToEulerAngles<double>(rot_poses);
            cv::Mat euler_angle_B = rotationMatrixToEulerAngles<double>(rot_pnp);
            std::cout << "Euler angle A: " << euler_angle_A << std::endl;
            std::cout << "Euler angle B: " << euler_angle_B << std::endl;

            std::cout << "AX: " << ax << std::endl;
            std::cout << "ZB: " << zb << std::endl;
            double trans_err, trans_err_noise, rot_error, rot_err_noise;
            translationError(ax, zb, trans_err);
            rotationError(ax, zb, rot_error);
            std::cout << "Trans error: " << trans_err << std::endl;
            std::cout << "Rot error: " << rot_error << std::endl;


            // NOISE
            double cov_trans = 0.1;
            double cov_rot = 0.01;

            cv::Mat noise_mat = cv::Mat::zeros(1, 1, rototras_vec[0][i].type());
            cv::randn(noise_mat, 0, cov_trans);
            double noise = noise_mat.at<double>(0,0);

            cv::Mat noise_mat_rot = cv::Mat::zeros(1, 1, rototras_vec[0][i].type());
            cv::randn(noise_mat_rot, 0, cov_rot);
            double noise_rot = noise_mat_rot.at<double>(0,0);

            std::cout << "Noise trans: " << noise << std::endl;
            std::cout << "Noise rot: " << noise_rot << std::endl;


            cv::Mat pnp = rototras_vec[0][i].inv();

            // Add translation noise
            poses_collected[0][i].at<double>(2,3) = poses_collected[0][i].at<double>(2,3) + noise;
            pnp.at<double>(0,3) = pnp.at<double>(0,3) - noise;

            getTras<double>(poses_collected[0][i], tras_poses);
            getTras<double>(pnp, tras_pnp);

            euler_angle_A.at<double>(0,1) = euler_angle_A.at<double>(0,1) + noise_rot;
            euler_angle_B.at<double>(0,1) = euler_angle_B.at<double>(0,1) - noise_rot;

            cv::Mat pose_rot_noise = eulerAnglesToRotationMatrix<double>(euler_angle_A);
            cv::Mat pnp_rot_noise = eulerAnglesToRotationMatrix<double>(euler_angle_B);
            cv::Mat pose_no, pnp_no;
            getRotoTras<double>(pose_rot_noise, tras_poses, pose_no);
            getRotoTras<double>(pnp_rot_noise, tras_pnp, pnp_no);

            poses_collected[0][i] = pose_no;
            pnp = pnp_no;
            std::cout << "Noise: " << std::endl;
            cv::Mat ax_noise = poses_collected[0][i] * gt1;
            cv::Mat zb_noise = b2e_gt * pnp;
            std::cout << "AX: " << ax_noise << std::endl;
            std::cout << "ZB: " << zb_noise << std::endl;

            translationError(ax_noise, zb_noise, trans_err_noise);
            rotationError(ax_noise, zb_noise, rot_err_noise);
            std::cout << "Trans error: " << trans_err_noise << std::endl;
            std::cout << "Rot error: " << rot_err_noise << std::endl;
        }
    }*/


    double trans1, trans2, trans3;
    double rot1, rot2, rot3;
    /*std::cout << "############### Evaluation: ############### " << std::endl;
    translationError(gt1, multi_h2e_optimal_vec[0].inv(), trans1);
    rotationError(gt1, multi_h2e_optimal_vec[1].inv(), rot1);
    std::cout << "Trans h2e 1: " << trans1 << " m"<< std::endl;
    std::cout << "Rot h2e 1: " << rot1 << " deg" << std::endl;

    translationError(gt2, multi_h2e_optimal_vec[1].inv(), trans2);
    rotationError(gt2, multi_h2e_optimal_vec[1].inv(), rot2);
    std::cout << "Trans h2e 2: " << trans2 <<" m"<< std::endl;
    std::cout << "Rot h2e 2: " << rot2 << " deg" << std::endl;

    translationError(gt3, multi_h2e_optimal_vec[1]*multi_h2e_optimal_vec[0].inv(), trans3);
    rotationError(gt3, multi_h2e_optimal_vec[1]*multi_h2e_optimal_vec[0].inv(), rot3);
    std::cout << "Trans cam2cam: " << trans3 <<" m"<< std::endl;
    std::cout << "Rot cam2cam: " << rot3 << " deg" << std::endl;*/



    /*std::cout << "############ GLOBAL OPTIMIZATOR ############" << std::endl;
    OptimizationData data;
    //cv::Mat h2e_opt = multi_h2e_optimal_vec[0].inv();
    //data.X_fixed = cvMatToEigen(h2e_opt); // Imposta i valori fissi per X
    //data.Z_fixed = cvMatToEigen(multi_b2ee_optimal); // Imposta i valori fissi per Z

    // Assicurati di impostare tz a un valore neutro, dato che verrà sovrascritto durante l'ottimizzazione
    //data.X_fixed(2, 3) = 0; // tz di X
    //data.Z_fixed(2, 3) = 0; // tz di Z
    int camera = 0;

    // Prepara i dati per la funzione obiettivo
    std::vector<std::pair<cv::Mat, cv::Mat>> data_opt;
    for (size_t i = 0; i < poses_collected[camera].size(); ++i) {
        if (cross_observation_matrix[i][camera] && i > 0) {
            //data_opt.emplace_back(poses_collected[camera][i], rototras_vec[camera][i].inv()); // T^{W}_{E} - T^{C}_{B}
            data_opt.emplace_back(poses_collected[camera][i-1].inv()*poses_collected[camera][i], rototras_vec[camera][i-1] * rototras_vec[camera][i].inv());
        }
    }

    data.transformPairs = data_opt;

    const int num_param = 6;
    // Configura NLopt
    nlopt::opt optimizer(nlopt::GN_DIRECT_L, num_param); // Esempio: utilizza l'algoritmo DIRECT

    optimizer.set_min_objective(handEyeObjectiveAXXB, &data);
    // Definisce i limiti inferiori e superiori per i parametri
    std::vector<double> lb(num_param );
    std::vector<double> ub(num_param );

    std::vector<double> x_glob_opt(num_param);

    x_glob_opt[0] = euler_mats[camera].at<double>(0,0);
    x_glob_opt[1] = euler_mats[camera].at<double>(0,1);
    x_glob_opt[2] = euler_mats[camera].at<double>(0,2);
    x_glob_opt[3] = tras_h2e_vec[camera].at<double>(0,0);
    x_glob_opt[4] = tras_h2e_vec[camera].at<double>(0,1);
    x_glob_opt[5] = tras_h2e_vec[camera].at<double>(0,0);

    x_glob_opt[6] = euler_b2e.at<double>(0,0);
    x_glob_opt[7] = euler_b2e.at<double>(0,1);
    x_glob_opt[8] = euler_b2e.at<double>(0,2);
    x_glob_opt[9] = b2e_tras.at<double>(0,0);
    x_glob_opt[10] = b2e_tras.at<double>(0,1);
    x_glob_opt[11] = b2e_tras.at<double>(0,2);

    lb[0] = euler_mats[camera].at<double>(0,0)-0.002;
    lb[1] = euler_mats[camera].at<double>(0,1)-0.002;
    lb[2] = euler_mats[camera].at<double>(0,2)-0.002;
    lb[3] = tras_h2e_vec[camera].at<double>(0,0)-0.02;
    lb[4] = tras_h2e_vec[camera].at<double>(0,1)-0.02;
    lb[5] = tras_h2e_vec[camera].at<double>(0,0) - 1;

    lb[6] = euler_b2e.at<double>(0,0)-0.002;
    lb[7] = euler_b2e.at<double>(0,1)-0.002;
    lb[8] = euler_b2e.at<double>(0,2)-0.002;
    lb[9] = b2e_tras.at<double>(0,0) -0.02;
    lb[10] = b2e_tras.at<double>(0,1) -0.02;
    lb[11] = b2e_tras.at<double>(0,2) - 1;

    ub[0] = euler_mats[camera].at<double>(0,0)+0.002;
    ub[1] = euler_mats[camera].at<double>(0,1)+0.002;
    ub[2] = euler_mats[camera].at<double>(0,2)+0.002;
    ub[3] = tras_h2e_vec[camera].at<double>(0,0)+0.02;
    ub[4] = tras_h2e_vec[camera].at<double>(0,1)+0.02;
    ub[5] = tras_h2e_vec[camera].at<double>(0,0) + 1;

    ub[6] = euler_b2e.at<double>(0,0)+0.002;
    ub[7] = euler_b2e.at<double>(0,1)+0.002;
    ub[8] = euler_b2e.at<double>(0,2)+0.002;
    ub[9] = b2e_tras.at<double>(0,0) +0.02;
    ub[10] = b2e_tras.at<double>(0,1) +0.02;
    ub[11] = b2e_tras.at<double>(0,2) + 1;


    optimizer.set_lower_bounds(lb);
    optimizer.set_upper_bounds(ub);
    int n = 10;
    //optimizer.set_maxeval(10000);

    // Esegui l'ottimizzazione
    cv::Mat temp = multi_h2e_optimal_vec[0].inv();

    double minf_opt;
    try {
        nlopt::result result = optimizer.optimize(x_glob_opt, minf_opt);
        std::cout << "Risultato ottimizzazione: " << minf_opt << std::endl;
        for (int i = 0; i < x_glob_opt.size(); i++) {
            std::cout << "Risultato X " << std::to_string(i) << ": " << x_glob_opt[i] << std::endl;
        }

    } catch (std::exception& e) {
        std::cerr << "Eccezione NLopt: " << e.what() << std::endl;
    }*/

    /*std::cout << "############ BAYES OPTIMIZATOR ############" << std::endl;
    int camera = 0;
    OptimizationDataBayes globalOptData;

    cv::Mat h2e_mat = multi_h2e_optimal_vec[0];
    globalOptData.X_fixed = cvMatToEigen(h2e_mat);
    //globalOptData.Z_fixed = cvMatToEigen(multi_b2ee_optimal);
    globalOptData.poses_A = poses_collected[0];
    globalOptData.poses_B = rototras_vec[0];
    globalOptData.cross_observation = cross_observation_matrix;
    std::vector<double> temp_xvec{h2e_mat.at<double>(2,3)};
    for (int i =0; i < temp_xvec.size(); i++) {
        std::cout << "Temp xvcec; " << temp_xvec[i] << std::endl;
    }

    // Configurazione iniziale di BayesOpt
    bayesopt::Parameters par = initialize_parameters_to_default();
    par.n_iterations = 10000;
    par.n_init_samples = 1000;
    par.kernel.name = "kSEISO";

    size_t dim = 1;
    vectord x_opt(dim); // Due parametri: tz per X e tz per Z
    x_opt[0] = h2e_mat.at<double>(2,3);
    //x_opt[1] = multi_b2ee_optimal.at<double>(2,3);

    // Imposta l'ottimizzatore
    HandEyeOptimization opt(&globalOptData, dim, par);
    vectord lowerBound(dim);
    vectord upperBound(dim);

    lowerBound[0] = h2e_mat.at<double>(2,3)-0.05;
    //lowerBound[1] = multi_b2ee_optimal.at<double>(2,3)-0.05;

    upperBound[0] = h2e_mat.at<double>(2,3)+0.5;
    //upperBound[1] = multi_b2ee_optimal.at<double>(2,3)+0.5;

    opt.setBoundingBox(lowerBound, upperBound);

    opt.optimize(x_opt);

    std::cout << "Parametri ottimizzati per X e Z: ";
    for (size_t i = 0; i < x_opt.size(); ++i) {
        std::cout << x_opt[i] << (i < x_opt.size() - 1 ? ", " : "\n");
    }*/



    // ANALYSIS NORMAL VECTORS
    /*std::vector<cv::Point3d> eigen_vecs_rob(3);
    std::vector<double> eigen_val_rob(3);
    std::vector<cv::Point3d> eigen_vecs_pnp(3);
    std::vector<double> eigen_val_pnp(3);
    for (int i = 0; i < 3; i++)
    {
        eigen_vecs_rob[i] = cv::Point3d(pca_robot.eigenvectors.at<double>(i, 0),
                                    pca_robot.eigenvectors.at<double>(i, 1),
                                    pca_robot.eigenvectors.at<double>(i, 2));
        eigen_val_rob[i] = pca_robot.eigenvalues.at<double>(i);
        std::cout <<"Eigen vec: " << eigen_vecs_rob[i] << std::endl;
        std::cout << "Eigen val: "<< eigen_val_rob[i] << std::endl;

        eigen_vecs_pnp[i] = cv::Point3d(pca_pnp.eigenvectors.at<double>(i, 0),
                                        pca_pnp.eigenvectors.at<double>(i, 1),
                                        pca_pnp.eigenvectors.at<double>(i, 2));
        eigen_val_pnp[i] = pca_pnp.eigenvalues.at<double>(i);
        std::cout <<"Eigen vec: " << eigen_vecs_pnp[i] << std::endl;
        std::cout << "Eigen val: "<< eigen_val_pnp[i] << std::endl;
    }

    cv::Mat h2e = multi_h2e_optimal_vec[0].inv();
    cv::Point3d cam = cv::Point3d(h2e.at<double>(0,3), h2e.at<double>(1,3), h2e.at<double>(2,3));
    double distance = dotProduct(eigen_vecs_rob[2], cam);
    std::cout <<"Distance from the equation of the plane: " << distance << std::endl;

    cv::Point3d base = cv::Point3d(multi_h2e_optimal_vec[0].at<double>(0,3), multi_h2e_optimal_vec[0].at<double>(1,3), multi_h2e_optimal_vec[0].at<double>(2,3));

    double distance_base = dotProduct(eigen_vecs_pnp[2], base);
    std::cout <<"Distance from the equation of the plane: " << distance_base << std::endl;


    double x_sum, y_sum, z_sum;
    for( int i = 0; i < poses_collected[0].size(); i++){
        x_sum += data_robot.at<double>(i,0);
        y_sum += data_robot.at<double>(i,1);
        z_sum += data_robot.at<double>(i,2);
    }
    x_sum = x_sum/poses_collected[0].size();
    y_sum = y_sum/poses_collected[0].size();
    z_sum = z_sum/poses_collected[0].size();

    cv::Point3d massPoint(x_sum, y_sum, z_sum);
    std::cout<<"Mass point: " << massPoint << std::endl;

    double var = dotProduct(eigen_vecs_rob[2], massPoint);
    double var2 = dotProduct(eigen_vecs_rob[2], massPoint);

    std::cout << "Var1: " << var << std::endl;
    std::cout << "Var2: " << var2 << std::endl;*/

}
