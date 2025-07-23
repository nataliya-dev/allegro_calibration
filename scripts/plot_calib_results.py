import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse


def load_calibration_results(json_file):
    """Load calibration results from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def matrix_from_json(matrix_data):
    """Convert JSON matrix data to numpy array."""
    return np.array(matrix_data['matrix'])


def plot_coordinate_frame(ax, transform_matrix, label, color='r', scale=0.1):
    """Plot a coordinate frame given a transformation matrix."""
    # Extract position
    position = transform_matrix[:3, 3]

    # Extract rotation matrix
    rotation = transform_matrix[:3, :3]

    # Define unit vectors for x, y, z axes
    axes = np.array([[scale, 0, 0],
                     [0, scale, 0],
                     [0, 0, scale]])

    # Transform axes to world coordinates
    world_axes = rotation @ axes.T

    # Plot the coordinate frame
    colors = ['red', 'green', 'blue']
    axis_labels = ['x', 'y', 'z']

    for i in range(3):
        ax.quiver(position[0], position[1], position[2],
                  world_axes[0, i], world_axes[1, i], world_axes[2, i],
                  color=colors[i], arrow_length_ratio=0.1, linewidth=2)

    # Add label
    ax.text(position[0], position[1], position[2] + 0.05, label,
            fontsize=10, color=color, weight='bold')

    return position


def visualize_camera_poses(json_file):
    """Visualize camera poses from calibration results."""
    # Load data
    data = load_calibration_results(json_file)

    # Create 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot robot base (origin)
    base_transform = np.eye(4)
    plot_coordinate_frame(ax, base_transform,
                          'Robot Base', 'black', scale=0.15)

    camera_positions = []
    base_position = np.array([0, 0, 0])  # Robot base at origin

    # Plot each camera
    for camera_name, camera_data in data['cameras'].items():
        if 'camera_to_robot' in camera_data:
            # Get transformation matrix
            transform = matrix_from_json(camera_data['camera_to_robot'])

            # Plot camera coordinate frame
            color = 'blue' if '1' in camera_name else 'cyan'
            position = plot_coordinate_frame(ax, transform, camera_name.replace('_', ' ').title(),
                                             color, scale=0.1)
            camera_positions.append(position)

            # Draw line from base to camera
            ax.plot([base_position[0], position[0]],
                    [base_position[1], position[1]],
                    [base_position[2], position[2]],
                    color=color, linestyle='--', alpha=0.7, linewidth=2)

            # Print camera position and orientation info
            print(f"\n{camera_name.replace('_', ' ').title()}:")
            print(
                f"  Position (m): [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")

            # Extract Euler angles (ZYX convention)
            R = transform[:3, :3]
            sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
            singular = sy < 1e-6

            if not singular:
                x = np.arctan2(R[2, 1], R[2, 2])
                y = np.arctan2(-R[2, 0], sy)
                z = np.arctan2(R[1, 0], R[0, 0])
            else:
                x = np.arctan2(-R[1, 2], R[1, 1])
                y = np.arctan2(-R[2, 0], sy)
                z = 0

            print(
                f"  Rotation (deg): Roll={np.degrees(x):.1f}, Pitch={np.degrees(y):.1f}, Yaw={np.degrees(z):.1f}")

    # Note: Board transformation not plotted as requested

    # Set reasonable axis limits to show relative positions clearly
    ax.set_xlim(-0.5, 1.0)
    ax.set_ylim(-1.0, 0.5)
    ax.set_zlim(0, 1.5)

    # Labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Camera Poses - {data["calibration_setup"].title()} Setup\n'
                 f'Number of cameras: {data["number_of_cameras"]}')

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='black', lw=2, label='Robot Base'),
        plt.Line2D([0], [0], color='blue', lw=2, label='Camera 1'),
        plt.Line2D([0], [0], color='cyan', lw=2, label='Camera 2'),
        plt.Line2D([0], [0], color='red', lw=1, label='X-axis'),
        plt.Line2D([0], [0], color='green', lw=1, label='Y-axis'),
        plt.Line2D([0], [0], color='blue', lw=1, label='Z-axis')
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Save the plot
    output_file = json_file.replace('.json', '_camera_poses.png')
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize camera poses from calibration results')
    parser.add_argument(
        'json_file', help='Path to the calibration results JSON file')

    args = parser.parse_args()

    try:
        visualize_camera_poses(args.json_file)
    except FileNotFoundError:
        print(f"Error: Could not find file '{args.json_file}'")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{args.json_file}'")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
