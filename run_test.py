import subprocess
import time

from test_config import feature_tracker_names_str
import glob

import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def umeyama_alignment(X, Y):
    """
    Umeyama algorithm for rigid alignment of two point sets.

    Parameters:
    X - a 3 x n matrix whose columns are points from camera trajectory
    Y - a 3 x n matrix whose columns are points from GPS trajectory

    Returns:
    c, R, t - the scaling, rotation matrix, and translation vector
    """
    m, n = X.shape

    # Calculate centroids
    mx = X.mean(axis=1, keepdims=True)
    my = Y.mean(axis=1, keepdims=True)

    # Center the points
    Xc = X - mx
    Yc = Y - my

    # Calculate variance of X
    sx = np.mean(np.sum(Xc * Xc, axis=0))

    # Calculate covariance matrix
    Sxy = np.dot(Yc, Xc.T) / n

    # SVD of covariance matrix
    U, D, Vt = np.linalg.svd(Sxy, full_matrices=True)
    V = Vt.T

    # Determine if we need to correct the rotation matrix
    S = np.eye(m)
    if np.linalg.det(Sxy) < 0:
        S[m - 1, m - 1] = -1

    # Calculate rotation matrix
    R = np.dot(U, np.dot(S, V.T))

    # Calculate scaling factor
    c = np.trace(np.dot(np.diag(D), S)) / sx

    # Calculate translation vector
    t = my - c * np.dot(R, mx)

    return c, R, t.flatten()


def calculate_ate(X, Y, c, R, t):
    """
    Calculate Absolute Trajectory Error (ATE) after alignment.

    Parameters:
    X - camera trajectory (3 x n matrix)
    Y - GPS trajectory (3 x n matrix)
    c, R, t - alignment parameters

    Returns:
    errors - array of errors for each point
    rmse - root mean square error
    """
    # Transform X to align with Y
    X_aligned = c * np.dot(R, X) + t.reshape(3, 1)

    # Calculate errors
    errors = np.sqrt(np.sum((Y - X_aligned) ** 2, axis=0))

    # Calculate RMSE
    rmse = np.sqrt(np.mean(errors ** 2))

    return errors, rmse


def calculate_rpe(X, Y, c, R, t, delta=1):
    """
    Calculate Relative Pose Error (RPE) after alignment.

    Parameters:
    X - camera trajectory (3 x n matrix)
    Y - GPS trajectory (3 x n matrix)
    c, R, t - alignment parameters
    delta - frame interval for calculating relative error

    Returns:
    rel_errors - array of relative errors
    rmse - root mean square error of relative errors
    """
    # Transform X to align with Y
    X_aligned = c * np.dot(R, X) + t.reshape(3, 1)

    n = X.shape[1]
    rel_errors = []

    for i in range(0, n - delta):
        # Calculate relative motion in both trajectories
        rel_X = X_aligned[:, i + delta] - X_aligned[:, i]
        rel_Y = Y[:, i + delta] - Y[:, i]

        # Calculate error
        error = np.sqrt(np.sum((rel_Y - rel_X) ** 2))
        rel_errors.append(error)

    rel_errors = np.array(rel_errors)
    rmse = np.sqrt(np.mean(rel_errors ** 2))

    return rel_errors, rmse


def synchronize_trajectories(camera_timestamps, camera_positions,
                             gps_timestamps, gps_positions, max_time_diff=0.1):
    """
    Synchronize camera and GPS trajectories based on timestamps.

    Parameters:
    camera_timestamps - array of camera timestamps
    camera_positions - 3 x n array of camera positions
    gps_timestamps - array of GPS timestamps
    gps_positions - 3 x m array of GPS positions
    max_time_diff - maximum allowed time difference for matching

    Returns:
    synced_camera_pos - synchronized camera positions
    synced_gps_pos - synchronized GPS positions
    """
    synced_camera = []
    synced_gps = []

    for i, cam_time in enumerate(camera_timestamps):
        # Find closest GPS timestamp
        time_diffs = np.abs(gps_timestamps - cam_time)
        min_idx = np.argmin(time_diffs)

        if time_diffs[min_idx] <= max_time_diff:
            synced_camera.append(camera_positions[:, i])
            synced_gps.append(gps_positions[:, min_idx])

    return np.array(synced_camera).T, np.array(synced_gps).T


# Example usage function
def evaluate_trajectories(camera_traj, gps_traj, visualization=True):
    """
    Complete trajectory evaluation pipeline.

    Parameters:
    camera_traj - 3 x n matrix of camera positions
    gps_traj - 3 x n matrix of GPS positions
    visualization - whether to plot results

    Returns:
    results - dictionary containing all error metrics
    """
    # Step 1: Align trajectories using Umeyama algorithm
    c, R, t = umeyama_alignment(camera_traj, gps_traj)

    # Step 2: Calculate ATE
    ate_errors, ate_rmse = calculate_ate(camera_traj, gps_traj, c, R, t)

    # Step 3: Calculate RPE
    rpe_errors, rpe_rmse = calculate_rpe(camera_traj, gps_traj, c, R, t, delta=1)

    # Step 4: Transform camera trajectory for visualization
    camera_traj_aligned = c * np.dot(R, camera_traj) + t.reshape(3, 1)

    results = {
        'alignment': {'scale': c, 'rotation': R, 'translation': t},
        'ate': {'errors': ate_errors, 'rmse': ate_rmse},
        'rpe': {'errors': rpe_errors, 'rmse': rpe_rmse},
        'aligned_camera_traj': camera_traj_aligned
    }

    if visualization:
        plot_results(camera_traj, gps_traj, camera_traj_aligned,
                     ate_errors, rpe_errors, ate_rmse, rpe_rmse)

    return results


def plot_results(camera_traj, gps_traj, camera_aligned,
                 ate_errors, rpe_errors, ate_rmse, rpe_rmse):
    """Plot trajectory comparison and error metrics."""
    fig = plt.figure(figsize=(15, 10))

    # Original trajectories
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(camera_traj[0], camera_traj[1], camera_traj[2], 'r-',
             label='Camera Trajectory', linewidth=2)
    ax1.plot(gps_traj[0], gps_traj[1], gps_traj[2], 'b-',
             label='GPS Trajectory', linewidth=2)
    ax1.set_title('Original Trajectories')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.legend()

    # Aligned trajectories
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.plot(camera_aligned[0], camera_aligned[1], camera_aligned[2], 'r-',
             label='Aligned Camera', linewidth=2)
    ax2.plot(gps_traj[0], gps_traj[1], gps_traj[2], 'b-',
             label='GPS Trajectory', linewidth=2)
    ax2.set_title('Aligned Trajectories')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.legend()

    # ATE plot
    ax3 = fig.add_subplot(223)
    ax3.plot(ate_errors, 'g-', linewidth=2)
    ax3.set_title(f'Absolute Trajectory Error\nRMSE: {ate_rmse:.4f} m')
    ax3.set_xlabel('Point Index')
    ax3.set_ylabel('Error (m)')
    ax3.grid(True)

    # RPE plot
    ax4 = fig.add_subplot(224)
    ax4.plot(rpe_errors, 'm-', linewidth=2)
    ax4.set_title(f'Relative Pose Error\nRMSE: {rpe_rmse:.4f} m')
    ax4.set_xlabel('Point Index')
    ax4.set_ylabel('Error (m)')
    ax4.grid(True)

    plt.tight_layout()
    plt.show()


# Load your data and run evaluation
# camera_traj = your_camera_data  # 3 x n matrix
# gps_traj = your_gps_data        # 3 x n matrix
# results = evaluate_trajectories(camera_traj, gps_traj)


arg1 = "--tracker_config"
mode = "VO"
for i in range(0,40):
    print(f'#######RUNNING TEST  {feature_tracker_names_str[i]} {i}  ###########################')
    if mode =="SLAM":
        subprocess.run(['python3', 'main_slam.py', arg1, str(i),"--headless"])
        time.sleep(10)
        log_files = glob.glob('MT_SLAM/logs/*')

        for log_file in log_files:
            if not os.path.basename(log_file).startswith('.'):  # Check if the file is not hidden
                try:
                    os.remove(log_file)
                    print(f'Removed log file: {log_file}')
                except Exception as e:
                   print(f'Error removing file {log_file}: {e}')
    if mode == "VO":
        subprocess.run(['python3', 'main_vo.py', arg1, str(i)])