#include <iostream>
#include "Calibrator.h"

int main(int argc, char* argv[]) {

    // Check if there are at least two arguments (program name + user argument)
    int number_of_waypoints = -1;
    int start_index = -1;
    std::string data_folder;
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <data_folder> && <number_of_images_to_use (-1 = all available images)>" << std::endl;
        return 1;
    }

    if (argc == 2) {
        // Get the user-specified argument from command line
        data_folder = argv[1];
        std::cout << "Folder: " << data_folder << std::endl;
    }

    if (argc == 3){
        data_folder = argv[1];
        std::cout << "Folder: " << data_folder << std::endl;
        try {
            number_of_waypoints = std::stoi(argv[2]);
        } catch (const std::invalid_argument& e) {
            std::cout << "Conversion failed: Invalid argument" << std::endl;
            return 1;
        } catch (const std::out_of_range& e) {
            std::cout << "Conversion failed: Out of range" << std::endl;
            return 1;
        }
    }

    if (argc == 4){
        data_folder = argv[1];
        std::cout << "Folder: " << data_folder << std::endl;
        try {
            number_of_waypoints = std::stoi(argv[2]);
        } catch (const std::invalid_argument& e) {
            std::cout << "Conversion failed: Invalid argument" << std::endl;
            return 1;
        } catch (const std::out_of_range& e) {
            std::cout << "Conversion failed: Out of range" << std::endl;
            return 1;
        }
        try {
            start_index = std::stoi(argv[3]);
        } catch (const std::invalid_argument& e) {
            std::cout << "Conversion failed: Invalid argument" << std::endl;
            return 1;
        } catch (const std::out_of_range& e) {
            std::cout << "Conversion failed: Out of range" << std::endl;
            return 1;
        }
    }

    if (argc > 4){
        std::cerr << "Too many input arguments!" << std::endl;
        return 1;
    }



    // Start the calibration
    Calibrator calibrator(data_folder, number_of_waypoints, start_index);
    calibrator.calibration();

    return 0;
}

