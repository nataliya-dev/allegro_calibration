import json
import numpy as np
import os
import csv
from scipy.spatial.transform import Rotation as R


def pose_to_transform_matrix(position, orientation):
    """
    Convert position [x, y, z] and quaternion [x, y, z, w] to 4x4 transformation matrix.
    """
    # Create rotation object from quaternion (scipy expects [x, y, z, w] format)
    rotation = R.from_quat(orientation)

    # Get rotation matrix
    rot_matrix = rotation.as_matrix()

    # Create 4x4 transformation matrix
    T = np.eye(4)
    T[0:3, 0:3] = rot_matrix
    T[0:3, 3] = position

    return T


def convert_poses_to_csv(input_file, output_folder='poses'):
    """
    Convert JSON poses to individual CSV files containing transformation matrices.
    """
    # Read the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process each sample
    for sample_name, sample_data in data.items():
        # Extract sample number from name (e.g., "sample_0000" -> "0000")
        sample_num = sample_name.split('_')[1]
        position = sample_data['position']
        orientation = sample_data['orientation']  # [x, y, z, w] format

        # Convert to transformation matrix
        T = pose_to_transform_matrix(position, orientation)

        # Save as CSV file
        output_file = os.path.join(output_folder, f'{sample_num}.csv')
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ')
            # Write all 4 rows of the transformation matrix
            for row in T:
                # Format numbers in scientific notation with high precision
                formatted_row = [f'{val:.18e}' for val in row]
                writer.writerow(formatted_row)

        print(f"Saved {output_file}")


# Example usage
if __name__ == "__main__":
    # Convert the first file
    convert_poses_to_csv('ee_poses.json', 'poses')

    # If you want to convert the second file to a different folder
    # convert_poses_to_csv('paste-2.txt', 'poses2')

    print("Conversion complete!")
