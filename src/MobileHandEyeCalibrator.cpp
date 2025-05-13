//
// Created by davide on 11/12/23.
//
#include "MobileHandEyeCalibrator.h"

MobileHandEyeCalibrator::MobileHandEyeCalibrator(const int number_of_waypoints, const int sens_quantity, const std::vector<cv::Point3f> object_points, std::vector<cv::Mat> poses, const std::vector<cv::Mat> h2e_initial_guess_vec, const cv::Mat b2ee_initial_guess, const std::vector<std::vector<int>> cross_observation_matrix, std::vector<std::vector<cv::Mat>> rvec_all, std::vector<std::vector<cv::Mat>> tvec_all, std::vector<std::vector<cv::Mat>> pnp, const std::vector<std::vector<double>> tz, const std::vector<std::vector<double>> cam_z, const std::vector<std::vector<cv::Mat>> relative_robot_poses, const std::vector<std::vector<cv::Mat>> relative_cam_poses) {
    number_of_cameras_ = sens_quantity;
    number_of_waypoints_ = number_of_waypoints;
    cross_observation_matrix_ = cross_observation_matrix;
    robot_r_vecs_.resize(number_of_waypoints);
    robot_t_vecs_.resize(number_of_waypoints);
    robot_r_vecs_inv_.resize(number_of_waypoints);
    robot_t_vecs_inv_.resize(number_of_waypoints);
    pnp_t_vecs_.resize(sens_quantity, std::vector<Eigen::Vector3d>(number_of_waypoints));
    pnp_r_vecs_.resize(sens_quantity, std::vector<Eigen::Vector3d>(number_of_waypoints));
    board_z_ = tz;
    cam_z_ = cam_z;


    robot_t_rel_.resize(sens_quantity, std::vector<Eigen::Vector3d>(number_of_waypoints-1));
    robot_r_rel_.resize(sens_quantity, std::vector<Eigen::Vector3d>(number_of_waypoints-1));
    pnp_t_rel_.resize(sens_quantity, std::vector<Eigen::Vector3d>(number_of_waypoints-1));
    pnp_r_rel_.resize(sens_quantity, std::vector<Eigen::Vector3d>(number_of_waypoints-1));


    // Array initialization
    h2e_ = new double*[number_of_cameras_];
    cam2cam_ = new double**[number_of_cameras_];
    for (int i = 0; i < number_of_cameras_; i++) {
        h2e_[i] = new double[6];
        cam2cam_[i] = new double*[number_of_cameras_];
        for (int j = 0; j < number_of_cameras_; j++) {
            cam2cam_[i][j] = new double[6];
        }
    }




    // H2e and Cam2cam definition
    for (int i = 0; i < number_of_cameras_; i++){
        cv::Mat_<double> r_vec_h2e, t_vec_h2e;
        transfMat2Exp<double>(h2e_initial_guess_vec[i], r_vec_h2e, t_vec_h2e);
        for( int j = 0; j < 3; j++ )
        {
            h2e_[i][j] = r_vec_h2e(j);
            h2e_[i][j+3] = t_vec_h2e(j);
        }


        for (int j = 0; j < number_of_cameras_; j++){
            if (i!=j){
                cv::Mat_<double> r_vec_c2c, t_vec_c2c;
                transfMat2Exp<double>(h2e_initial_guess_vec[i]*h2e_initial_guess_vec[j].inv(), r_vec_c2c, t_vec_c2c); // j-th camera frame in i-th camera frame
                for (int k = 0; k < 3; k++){

                    cam2cam_[i][j][k] = r_vec_c2c(k);
                    cam2cam_[i][j][k+3] = t_vec_c2c(k);
                }
            }
        }
    }

    // Save tvec and rvec for each robot path_pose
    for (int wp = 0; wp < number_of_waypoints_; wp++) {
        cv::Mat_<double> r_vec_w2e, t_vec_w2e;
        cv::Mat_<double> rel_r_vec_w2e, rel_t_vec_w2e;

        cv::Mat robot_pose_temp = poses[wp];
        transfMat2Exp<double>(robot_pose_temp, r_vec_w2e, t_vec_w2e);

        for (int j = 0; j < 3; j++) {
            robot_r_vecs_[wp](j) = r_vec_w2e(j);
            robot_t_vecs_[wp](j) = t_vec_w2e(j);
        }
    }


    // RELATIVE POSES
    for (int i = 0; i < number_of_cameras_; i++) {
        for (int wp = 0; wp < number_of_waypoints_ - 1; wp++) {
            cv::Mat_<double> rel_r_vec_w2r, rel_t_vec_w2r;

            cv::Mat robot_rel_pose = relative_robot_poses[i][wp];
            if (!robot_rel_pose.empty()) {
                transfMat2Exp<double>(robot_rel_pose, rel_r_vec_w2r, rel_t_vec_w2r);

                for (int j = 0; j < 3; j++) {
                    robot_r_rel_[i][wp](j) = rel_r_vec_w2r(j);
                    robot_t_rel_[i][wp](j) = rel_t_vec_w2r(j);
                }
            }
        }
    }

    for (int i = 0; i < number_of_cameras_; i++){
        for (int wp = 0; wp < number_of_waypoints_-1; wp++){

            cv::Mat_<double> rel_r_vec_c2b, rel_t_vec_c2b;

            cv::Mat cam_rel_pose = relative_cam_poses[i][wp];
            if (!cam_rel_pose.empty()){
                transfMat2Exp<double>(cam_rel_pose, rel_r_vec_c2b, rel_t_vec_c2b);
                for (int j = 0; j < 3; j++) {
                    pnp_r_rel_[i][wp](j) = rel_r_vec_c2b(j);
                    pnp_t_rel_[i][wp](j) = rel_t_vec_c2b(j);
                }
            }

        }
    }



    for (int i = 0; i < number_of_cameras_; i++){
        for (int wp = 0; wp < number_of_waypoints_; wp++){
            if (cross_observation_matrix_[wp][i]) {
                //std::cout << "Pnp " << std::to_string(wp) << ": " << tvec_all[i][wp] << std::endl;
                for (int j = 0; j < 3; j++) {
                    pnp_r_vecs_[i][wp](j) = rvec_all[i][wp].at<double>(j);
                    pnp_t_vecs_[i][wp](j) = tvec_all[i][wp].at<double>(j);
                }
            }
        }
    }



    cv::Mat_<double> r_vec, t_vec;
    transfMat2Exp<double>(b2ee_initial_guess, r_vec, t_vec);
    for( int j = 0; j < 3; j++ ){

        board2ee_[j] = r_vec(j);
        board2ee_[j+3] = t_vec(j);
    }

}

void MobileHandEyeCalibrator::mobileCalibration(const std::vector<CameraInfo> camera_info, std::vector<std::vector<std::vector<cv::Point2f>>> corners, std::vector<cv::Mat> &optimal_h2e, cv::Mat &optimal_b2ee, std::vector<std::vector<cv::Mat>> &optimal_cam2cam, std::vector<cv::Mat>& optim_poses_collected, std::vector<std::vector<cv::Mat>>& optim_pnp_collected) {
    double observed_pt_data[2];
    double observed_pt_data_dir[2];
    Eigen::Map<Eigen::Vector2d> observed_pt(observed_pt_data);
    Eigen::Map<Eigen::Vector2d> observed_pt_dir(observed_pt_data_dir);

    // Ceres problem
    ceres::Solver::Options options;

    // Minimizer type (TRUST_REGION or LINE_SEARCH)
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    //options.preconditioner_type = ceres::SCHUR_JACOBI;


    // Enabling CUDA optimization
    //options.dense_linear_algebra_library_type = ceres::CUDA;
    
    options.linear_solver_type = ceres::DENSE_SCHUR;  //ITERATIVE_SCHUR
    options.max_num_refinement_iterations = 3;
    options.use_mixed_precision_solves = true;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 20;
    options.max_num_iterations = 2000;
    options.gradient_tolerance = 1e-5;  // 1e-5
    options.function_tolerance = std::numeric_limits<double>::epsilon();
    options.parameter_tolerance = 1e-8;  // 1e-8
    bool enable_cauchy_loss = true;
    double humer_a_scale = 1;
    ceres::LossFunction* loss_function = new ceres::CauchyLoss(humer_a_scale);

    ceres::Problem problem_multi_mobile;

    std::vector<PinholeCameraModel> cam_model_vec;
    for (int i = 0; i < number_of_cameras_; i++){
        PinholeCameraModel camera_model(camera_info[i]);
        cam_model_vec.push_back(camera_model);
    }
    // Prepare the optimization algorithm
    for (int wp = 0; wp < number_of_waypoints_; wp++){

        showProgressBar(wp, number_of_waypoints_);

        // Check how many cameras are detecting
        int detection_amount = accumulate(cross_observation_matrix_[wp].begin(), cross_observation_matrix_[wp].end(), 0);

        // Single camera optimization problem whenever the checkerboard is detected
        if (detection_amount > 0){

            for (int cam = 0; cam < number_of_cameras_; cam++){
                // Check if the current camera detected the checkerboard in the current pose
                if (cross_observation_matrix_[wp][cam]) {

                    if (wp > 0) {
                        if (cross_observation_matrix_[wp - 1][cam]) {
                            if (cam_z_[cam][wp] == 0) {
                                /*ceres::CostFunction *cost_function_ax_xb_wo_z = Classic_ax_xb_wo_z::Create(
                                        pnp_r_vecs_[cam][wp - 1], pnp_t_vecs_[cam][wp - 1], robot_r_vecs_[wp - 1],
                                        robot_t_vecs_[wp - 1],
                                        pnp_r_vecs_[cam][wp], pnp_t_vecs_[cam][wp], robot_r_vecs_[wp],
                                        robot_t_vecs_[wp]);
                                problem_multi_mobile.AddResidualBlock(cost_function_ax_xb_wo_z, loss_function,
                                                                      h2e_[cam]);
*/
                                ceres::CostFunction *cost_function_ax_xb_wo_z_rel = Classic_ax_xb_wo_z_rel::Create(
                                        pnp_r_rel_[cam][wp - 1], pnp_t_rel_[cam][wp - 1], robot_r_rel_[cam][wp - 1],
                                        robot_t_rel_[cam][wp - 1]);
                                problem_multi_mobile.AddResidualBlock(cost_function_ax_xb_wo_z_rel, loss_function,
                                                                      h2e_[cam]);
                            } else {


                               /*ceres::CostFunction *cost_function_ax_xb = Classic_ax_xb::Create(
                                        pnp_r_vecs_[cam][wp - 1], pnp_t_vecs_[cam][wp - 1], robot_r_vecs_[wp - 1],
                                        robot_t_vecs_[wp - 1],
                                        pnp_r_vecs_[cam][wp], pnp_t_vecs_[cam][wp], robot_r_vecs_[wp],
                                        robot_t_vecs_[wp], board_z_[cam][wp], cam_z_[cam][wp]);
                                problem_multi_mobile.AddResidualBlock(cost_function_ax_xb, loss_function, h2e_[cam]);*/

                                ceres::CostFunction *cost_function_ax_xb_rel = Classic_ax_xb_rel::Create(
                                        pnp_r_rel_[cam][wp - 1], pnp_t_rel_[cam][wp - 1], robot_r_rel_[cam][wp - 1],
                                        robot_t_rel_[cam][wp - 1], robot_r_vecs_[wp],
                                        robot_t_vecs_[wp], cam_z_[cam][wp]);
                                problem_multi_mobile.AddResidualBlock(cost_function_ax_xb_rel, loss_function, h2e_[cam]);

                                /*ceres::CostFunction *cost_function_ax_xb_opt = ax_xb_opt::Create(
                                        pnp_r_vecs_[cam][wp - 1], pnp_t_vecs_[cam][wp - 1], robot_r_vecs_[wp - 1],
                                        robot_t_vecs_[wp - 1],
                                        pnp_r_vecs_[cam][wp], pnp_t_vecs_[cam][wp], robot_r_vecs_[wp],
                                        robot_t_vecs_[wp], cam_z_[cam][wp]);
                                problem_multi_mobile.AddResidualBlock(cost_function_ax_xb_opt, loss_function, h2e_[cam], robot_opt_[wp]);*/
                            }
                        }
                    }

                    /*if (wp > 0) {
                        if (cross_observation_matrix_[wp - 1][cam]) {
                            ceres::CostFunction *cost_function_ax_xb = AX_XB::Create(
                                    pnp_r_vecs_[cam][wp - 1], pnp_t_vecs_[cam][wp - 1], robot_r_vecs_[wp - 1],
                                    robot_t_vecs_[wp - 1],
                                    pnp_r_vecs_[cam][wp], pnp_t_vecs_[cam][wp], robot_r_vecs_[wp],
                                    robot_t_vecs_[wp]);
                            problem_multi_mobile.AddResidualBlock(cost_function_ax_xb, loss_function, h2e_[cam]);
                        }
                    }*/


                }
            }

            // Multi camera optimization problem whenever the checkerboard is detected simultaneously by more cameras
            if (detection_amount>1){
                for (int cam_1 = 0; cam_1 < number_of_cameras_; cam_1++){
                    for (int cam_2 = 0; cam_2 < number_of_cameras_; cam_2++){

                        // Check that the cameras which are detecting are different
                        if (cam_1 != cam_2 && cross_observation_matrix_[wp][cam_1] && cross_observation_matrix_[wp][cam_2]){

                            if (wp > 0) {
                                /*if (cross_observation_matrix_[wp - 1][cam_1] && cross_observation_matrix_[wp - 1][cam_1] && cross_observation_matrix_[wp - 1][cam_2] && cross_observation_matrix_[wp - 1][cam_2]) {
                                    ceres::CostFunction *cost_function_ax_xb_rel = Classic_ax_xb_rel_multi::Create(
                                            pnp_r_rel_[cam_1][wp - 1], pnp_t_rel_[cam_1][wp - 1], pnp_r_rel_[cam_2][wp - 1], pnp_t_rel_[cam_2][wp - 1]);
                                    problem_multi_mobile.AddResidualBlock(cost_function_ax_xb_rel, loss_function, h2e_[cam_1], h2e_[cam_2]);
                                }*/
                            }
                        }
                    }
                }
            }
        }
    }

    ceres::Solver::Summary summary_cam2cam;
    Solve(options, &problem_multi_mobile, &summary_cam2cam);

    std::cout << summary_cam2cam.FullReport() << "\n";

    for (int cam = 0; cam < number_of_cameras_; cam++){
        cv::Mat optim_h2e_rvec = (cv::Mat_<double>(3,1) << h2e_[cam][0], h2e_[cam][1], h2e_[cam][2]);
        cv::Mat optim_h2e_tvec = (cv::Mat_<double>(3,1) << h2e_[cam][3], h2e_[cam][4], h2e_[cam][5]);
        cv::Mat optim_h2e_mat;

        exp2TransfMat<double>(optim_h2e_rvec, optim_h2e_tvec, optim_h2e_mat);
        optimal_h2e[cam] = optim_h2e_mat;
    }

    cv::Mat optim_board2tcp_rvec = (cv::Mat_<double>(3,1) << board2ee_[0], board2ee_[1], board2ee_[2]);
    cv::Mat optim_board2tcp_tvec = (cv::Mat_<double>(3,1) << board2ee_[3], board2ee_[4], board2ee_[5]);
    exp2TransfMat<double>(optim_board2tcp_rvec, optim_board2tcp_tvec, optimal_b2ee);


    for (int j = 0; j < number_of_cameras_; j++) {
        for (int i = 0; i < number_of_cameras_; i++) {

            cv::Mat optim_c2c_rvec = (cv::Mat_<double>(3, 1)
                    << cam2cam_[j][i][0], cam2cam_[j][i][1], cam2cam_[j][i][2]);
            cv::Mat optim_c2c_tvec = (cv::Mat_<double>(3, 1)
                    << cam2cam_[j][i][3], cam2cam_[j][i][4], cam2cam_[j][i][5]);
            cv::Mat optimal_c2c;
            exp2TransfMat<double>(optim_c2c_rvec, optim_c2c_tvec, optimal_c2c);
            //std::cout << "Cam " << std::to_string(i + 1) << " to " << std::to_string(j + 1) << ": " << optimal_c2c
            //          << std::endl;
        }
    }
}



void MobileHandEyeCalibrator::mobileJointCalibration(const std::vector<CameraInfo> camera_info, std::vector<std::vector<std::vector<cv::Point2f>>> corners, std::vector<cv::Mat> &optimal_h2e, cv::Mat &optimal_b2ee, std::vector<std::vector<cv::Mat>> &optimal_cam2cam, std::vector<cv::Mat>& optim_poses_collected, std::vector<std::vector<cv::Mat>>& optim_pnp_collected) {
    double observed_pt_data[2];
    double observed_pt_data_dir[2];
    Eigen::Map<Eigen::Vector2d> observed_pt(observed_pt_data);
    Eigen::Map<Eigen::Vector2d> observed_pt_dir(observed_pt_data_dir);

    // Ceres problem
    ceres::Solver::Options options;

    // Minimizer type (TRUST_REGION or LINE_SEARCH)
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    //options.preconditioner_type = ceres::SCHUR_JACOBI;


    // Enabling CUDA optimization
    //options.dense_linear_algebra_library_type = ceres::CUDA;

    options.linear_solver_type = ceres::DENSE_SCHUR;  //ITERATIVE_SCHUR
    options.max_num_refinement_iterations = 3;
    options.use_mixed_precision_solves = true;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 20;
    options.max_num_iterations = 2000;
    options.gradient_tolerance = 1e-5;  // 1e-5
    options.function_tolerance = std::numeric_limits<double>::epsilon();
    options.parameter_tolerance = 1e-8;  // 1e-8
    bool enable_cauchy_loss = true;
    double humer_a_scale = 1;
    ceres::LossFunction* loss_function = new ceres::CauchyLoss(humer_a_scale);

    ceres::Problem problem_multi_mobile;

    std::vector<PinholeCameraModel> cam_model_vec;
    for (int i = 0; i < number_of_cameras_; i++){
        PinholeCameraModel camera_model(camera_info[i]);
        cam_model_vec.push_back(camera_model);
    }
    // Prepare the optimization algorithm
    for (int wp = 0; wp < number_of_waypoints_; wp++){

        showProgressBar(wp, number_of_waypoints_);

        // Check how many cameras are detecting
        int detection_amount = accumulate(cross_observation_matrix_[wp].begin(), cross_observation_matrix_[wp].end(), 0);

        // Single camera optimization problem whenever the checkerboard is detected
        if (detection_amount > 0){

            for (int cam = 0; cam < number_of_cameras_; cam++){
                // Check if the current camera detected the checkerboard in the current pose
                if (cross_observation_matrix_[wp][cam]) {

                    if (wp > 0) {
                        if (cross_observation_matrix_[wp - 1][cam]) {
                            if (cam_z_[cam][wp] == 0) {
                                /*ceres::CostFunction *cost_function_ax_xb_wo_z = Classic_ax_xb_wo_z::Create(
                                        pnp_r_vecs_[cam][wp - 1], pnp_t_vecs_[cam][wp - 1], robot_r_vecs_[wp - 1],
                                        robot_t_vecs_[wp - 1],
                                        pnp_r_vecs_[cam][wp], pnp_t_vecs_[cam][wp], robot_r_vecs_[wp],
                                        robot_t_vecs_[wp]);
                                problem_multi_mobile.AddResidualBlock(cost_function_ax_xb_wo_z, loss_function,
                                                                      h2e_[cam]);
*/
                                ceres::CostFunction *cost_function_ax_xb_wo_z_rel = Classic_ax_xb_wo_z_rel::Create(
                                        pnp_r_rel_[cam][wp - 1], pnp_t_rel_[cam][wp - 1], robot_r_rel_[cam][wp - 1],
                                        robot_t_rel_[cam][wp - 1]);
                                problem_multi_mobile.AddResidualBlock(cost_function_ax_xb_wo_z_rel, loss_function,
                                                                      h2e_[cam]);
                            } else {


                                /*ceres::CostFunction *cost_function_ax_xb = Classic_ax_xb::Create(
                                         pnp_r_vecs_[cam][wp - 1], pnp_t_vecs_[cam][wp - 1], robot_r_vecs_[wp - 1],
                                         robot_t_vecs_[wp - 1],
                                         pnp_r_vecs_[cam][wp], pnp_t_vecs_[cam][wp], robot_r_vecs_[wp],
                                         robot_t_vecs_[wp], board_z_[cam][wp], cam_z_[cam][wp]);
                                 problem_multi_mobile.AddResidualBlock(cost_function_ax_xb, loss_function, h2e_[cam]);*/

                                ceres::CostFunction *cost_function_ax_xb_rel = Classic_ax_xb_rel::Create(
                                        pnp_r_rel_[cam][wp - 1], pnp_t_rel_[cam][wp - 1], robot_r_rel_[cam][wp - 1],
                                        robot_t_rel_[cam][wp - 1], robot_r_vecs_[wp],
                                        robot_t_vecs_[wp], cam_z_[cam][wp]);
                                problem_multi_mobile.AddResidualBlock(cost_function_ax_xb_rel, loss_function, h2e_[cam]);

                                /*ceres::CostFunction *cost_function_ax_xb_opt = ax_xb_opt::Create(
                                        pnp_r_vecs_[cam][wp - 1], pnp_t_vecs_[cam][wp - 1], robot_r_vecs_[wp - 1],
                                        robot_t_vecs_[wp - 1],
                                        pnp_r_vecs_[cam][wp], pnp_t_vecs_[cam][wp], robot_r_vecs_[wp],
                                        robot_t_vecs_[wp], cam_z_[cam][wp]);
                                problem_multi_mobile.AddResidualBlock(cost_function_ax_xb_opt, loss_function, h2e_[cam], robot_opt_[wp]);*/
                            }
                        }
                    }

                    /*if (wp > 0) {
                        if (cross_observation_matrix_[wp - 1][cam]) {
                            ceres::CostFunction *cost_function_ax_xb = AX_XB::Create(
                                    pnp_r_vecs_[cam][wp - 1], pnp_t_vecs_[cam][wp - 1], robot_r_vecs_[wp - 1],
                                    robot_t_vecs_[wp - 1],
                                    pnp_r_vecs_[cam][wp], pnp_t_vecs_[cam][wp], robot_r_vecs_[wp],
                                    robot_t_vecs_[wp]);
                            problem_multi_mobile.AddResidualBlock(cost_function_ax_xb, loss_function, h2e_[cam]);
                        }
                    }*/


                }
            }

            // Multi camera optimization problem whenever the checkerboard is detected simultaneously by more cameras
            if (detection_amount>1){
                for (int cam_1 = 0; cam_1 < number_of_cameras_; cam_1++){
                    for (int cam_2 = 0; cam_2 < number_of_cameras_; cam_2++){

                        // Check that the cameras which are detecting are different
                        if (cam_1 != cam_2 && cross_observation_matrix_[wp][cam_1] && cross_observation_matrix_[wp][cam_2]){

                            if (wp > 0) {
                                if (cross_observation_matrix_[wp - 1][cam_1] && cross_observation_matrix_[wp - 1][cam_1] && cross_observation_matrix_[wp - 1][cam_2] && cross_observation_matrix_[wp - 1][cam_2]) {
                                    ceres::CostFunction *cost_function_ax_xb_rel = Classic_ax_xb_rel_multi::Create(
                                            pnp_r_rel_[cam_1][wp - 1], pnp_t_rel_[cam_1][wp - 1], pnp_r_rel_[cam_2][wp - 1], pnp_t_rel_[cam_2][wp - 1]);
                                    problem_multi_mobile.AddResidualBlock(cost_function_ax_xb_rel, loss_function, h2e_[cam_1], h2e_[cam_2]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    ceres::Solver::Summary summary_cam2cam;
    Solve(options, &problem_multi_mobile, &summary_cam2cam);

    std::cout << summary_cam2cam.FullReport() << "\n";

    for (int cam = 0; cam < number_of_cameras_; cam++){
        cv::Mat optim_h2e_rvec = (cv::Mat_<double>(3,1) << h2e_[cam][0], h2e_[cam][1], h2e_[cam][2]);
        cv::Mat optim_h2e_tvec = (cv::Mat_<double>(3,1) << h2e_[cam][3], h2e_[cam][4], h2e_[cam][5]);
        cv::Mat optim_h2e_mat;

        exp2TransfMat<double>(optim_h2e_rvec, optim_h2e_tvec, optim_h2e_mat);
        optimal_h2e[cam] = optim_h2e_mat;
    }

    cv::Mat optim_board2tcp_rvec = (cv::Mat_<double>(3,1) << board2ee_[0], board2ee_[1], board2ee_[2]);
    cv::Mat optim_board2tcp_tvec = (cv::Mat_<double>(3,1) << board2ee_[3], board2ee_[4], board2ee_[5]);
    exp2TransfMat<double>(optim_board2tcp_rvec, optim_board2tcp_tvec, optimal_b2ee);


    for (int j = 0; j < number_of_cameras_; j++) {
        for (int i = 0; i < number_of_cameras_; i++) {

            cv::Mat optim_c2c_rvec = (cv::Mat_<double>(3, 1)
                    << cam2cam_[j][i][0], cam2cam_[j][i][1], cam2cam_[j][i][2]);
            cv::Mat optim_c2c_tvec = (cv::Mat_<double>(3, 1)
                    << cam2cam_[j][i][3], cam2cam_[j][i][4], cam2cam_[j][i][5]);
            cv::Mat optimal_c2c;
            exp2TransfMat<double>(optim_c2c_rvec, optim_c2c_tvec, optimal_c2c);
            //std::cout << "Cam " << std::to_string(i + 1) << " to " << std::to_string(j + 1) << ": " << optimal_c2c
            //          << std::endl;
        }
    }
}



