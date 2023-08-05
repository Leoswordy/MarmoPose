import os
import toml
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from glob import glob
from typing import Dict, Any, List, Tuple

from aniposelib.boards import merge_rows
from marmopose.utils.io import load_pose_2d, load_all_poses_2d


def plot_calibration(config: dict, n_show: int=1) -> None:
    """
    Plots the result of calibration for `n_show` randomly chosen frames.

    This function loads the previously saved calibration results and the detected boards from each camera frame. For
    each randomly chosen frame, it creates a two-column subplot. The first column shows the original image with 
    detected corners and the second column shows the calibrated image.

    Args:
        config: Configuration dictionary containing project settings.
        n_show: Number of frames to be shown. Default is 1.

    Raises:
        FileNotFoundError: If the required files are not found in the specified directory.
    """
    project_dir = config['project_dir']
    calibration_dir = config['directory']['calibration']
    result_dir = os.path.join(project_dir, calibration_dir)

    # Load detected board corners and the camera calibration results.
    rows_filepath = os.path.join(result_dir, 'detected_boards.pickle')
    with open(rows_filepath, 'rb') as f:
        all_rows = pickle.load(f)
    n_cam = len(all_rows)
    
    calibration_filepath = os.path.join(result_dir, 'calibration.toml')
    calibration_dict = toml.load(calibration_filepath)

    # Select only those frames where all cameras have detected the board.
    frame_board = [frame for frame in merge_rows(all_rows) if len(frame)==n_cam]
    show_idx = np.random.choice(len(frame_board), n_show)

    for idx in show_idx:
        plt.figure(figsize=(14, 14))
        for cam_id in range(1, n_cam+1):
            # Load camera parameters.
            detected_board = frame_board[idx][cam_id-1]
            K = np.array(calibration_dict[f'cam_{cam_id-1}']['matrix'])
            D = np.array(calibration_dict[f'cam_{cam_id-1}']['distortions'])
            size = calibration_dict[f'cam_{cam_id-1}']['size']

            # Load the corresponding image from the video.
            video_name = os.path.join(result_dir, f'cam{cam_id}.mp4')
            cap = cv2.VideoCapture(video_name)
            cap.set(cv2.CAP_PROP_POS_FRAMES, detected_board['framenum'][1])
            ret, image = cap.read()
            if ret:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Add detected corners to the image.
            corners_subpix = detected_board['corners']
            for corner in corners_subpix.reshape(-1, 2):
                image_corners = cv2.circle(np.zeros_like(image), (int(corner[0]),int(corner[1])), 15, [0, 255, 0], -1)
                image = cv2.add(image, image_corners)

            # Undistort the image using the camera parameters.
            K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, size, np.eye(3), balance=1.0)  
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_new, size, cv2.CV_32FC1)
            undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            plt.subplot(n_cam, 2, 2*cam_id-1)
            plt.imshow(image)
            plt.title(f"cam{cam_id} - frame{detected_board['framenum'][1]}")
            plt.subplot(n_cam, 2, 2*cam_id)
            plt.imshow(undistorted_image)
            plt.title(f'calibrated')
        plt.show()


def plot_scores(config: Dict[str, Any]) -> None:
    """Plots the scores of every bodypart across frames in each camera.

    This function iterates over all .h5 files located in the poses_2d directory. 
    For each file, it loads 2D poses and plots a histogram of scores for each 
    bodypart across all frames for each track.

    Args:
        config: A dictionary that includes various project configurations.
            It should have at least 'project_dir' key specifying the path to the project directory, 
            and 'directory' key, which is another dictionary that contains the name of 'poses_2d' directory.
    """
    project_dir = config['project_dir']
    poses_2d_dir = config['directory']['poses_2d']
    labels_list = sorted(glob(os.path.join(project_dir, poses_2d_dir, '*.h5')))

    for labels_fname in labels_list:
        base_name = os.path.basename(labels_fname)
        print(f'Plot file: {base_name}')
        plt.figure(figsize=(14, 14))
        points, metadata = load_pose_2d(labels_fname)
        n_tracks, n_frames, n_bodyparts, _ = points.shape
        bodyparts = metadata['bodyparts']
        tracks = metadata['tracks']

        for track_idx in range(n_tracks):
            for bp_idx, bp in enumerate(bodyparts):
                plt.subplot(4, 4, 1+bp_idx+track_idx*n_bodyparts)
                plt.title(f'{tracks[track_idx]} - {bp}')
                plt.hist(points[track_idx, :, bp_idx, 2], bins=40, density=True)
        plt.show()


def plot_visibility(config: Dict[str, Any], track_idx: int = 0, mode: str = 'each') -> None:
    """Plots the visibility of every bodypart across frames.

    Args:
        config: A dictionary that includes various project configurations.
        track_idx: The index of the target track to plot. Defaults to 0.
        mode: Plot mode. If 'each', plots the visibility of each camera. 
            If 'count', plots the number of cameras each bodypart is visible in. Defaults to 'each'.
    """
    project_dir = config['project_dir']
    poses_2d_dir = config['directory']['poses_2d']
    labels_list = sorted(glob(os.path.join(project_dir, poses_2d_dir, '*.h5')))
    bodyparts = config['bodyparts']

    all_points, metadata = load_all_poses_2d(labels_list) #(n_cams, n_tracks, n_frames, n_bodyparts, [x, y, scores])

    data = all_points[:, track_idx, :, :, 0]
    visible_mask = ~np.isnan(data)

    if mode == 'each':
        visible_ratio = np.mean(visible_mask, axis=1)

        df = pd.DataFrame(visible_ratio.T, columns=[f'cam{i+1}' for i in range(data.shape[0])])
        df['BodyParts'] = bodyparts
        df_melted = df.melt(id_vars='BodyParts', var_name='Camera', value_name='VisibleRatio')

        plt.figure(figsize=(15,8))
        sns.barplot(x='BodyParts', y='VisibleRatio', hue='Camera', data=df_melted, palette='viridis')
        plt.legend(loc='upper left', title='Camera')
        plt.title(f'Visible Ratio for Each Body Part in Every Camera (Frames={data.shape[1]})')
        plt.ylabel('Visible Ratio across Frames')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    if mode == 'count':
        visible_counts = np.sum(visible_mask, axis=0) # Count of visible cameras for each frame and body part

        bins = np.zeros((data.shape[0]+1, data.shape[2]))
        for i in range(data.shape[2]):
            bins[:, i] = np.bincount(visible_counts[:, i], minlength=data.shape[0]+1)
        ratios = bins / np.sum(bins, axis=0)

        df = pd.DataFrame(ratios.T, columns=[f'{i}' for i in range(data.shape[0]+1)])
        df['BodyParts'] = bodyparts
        df_melted = df.melt(id_vars='BodyParts', var_name='Number of Visible Camera', value_name='Ratio')

        blue_colors = ["#4287f5", "#3062d7", "#1f3fb9"]
        red_colors = ["#FF9999", "#FF6666"]
        sns.set_palette(red_colors + blue_colors)

        plt.figure(figsize=(15,8))
        sns.barplot(x='BodyParts', y='Ratio', hue='Number of Visible Camera', data=df_melted)
        plt.legend(loc='upper left', title='Number of Visible Camera')
        plt.title(f'Ratio of the Number of Visible Camera for Each Body Part (Frames={data.shape[1]})')
        plt.ylabel('Ratio across Frames')
        plt.xticks(rotation=45) 
        plt.tight_layout() 
        plt.show()
    

def plot_triangulation_errors(config: Dict[str, Any]) -> None:
    """Plots triangulation errors for bodyparts visible in more than one camera.

    Args:
        config: A dictionary that includes various project configurations.
    """
    def load_pose_3d_with_ncams(filepaths: List[str], 
                                bodyparts: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Loads 3D pose data along with the corresponding scores and errors.

        Args:
            filepaths: List of file paths to load data from.
            bodyparts: List of bodyparts to extract data for.

        Returns:
            1. All 3D points with shape of (n_tracks, n_bodyparts, n_frames, [x, y, z] coordinates).
            2. Score data with shape of (n_tracks, n_bodyparts, n_frames).
            3. Error data with shape of (n_tracks, n_bodyparts, n_frames).
            4. Data about the number of cameras each bodypart is visible in, with shape of (n_tracks, n_bodyparts, n_frames).
        """
        all_points, all_scores, all_errors  = [], [], []
        all_n_cams = []

        for file in filepaths:
            data = pd.read_csv(file)
            points, scores, errors = [], [], []
            n_cams = []

            for bp in bodyparts:
                points.append(data.loc[:, (bp+'_x', bp+'_y', bp+'_z')].values)
                scores.append(data.loc[:, bp+'_score'].values)
                errors.append(data.loc[:, bp+'_error'].values)
                n_cams.append(data.loc[:, bp+'_ncams'].values)

            all_points.append(points)
            all_scores.append(scores)
            all_errors.append(errors)
            all_n_cams.append(n_cams)

        return (np.array(all_points, dtype='float64'), np.array(all_scores, dtype='float64'), 
                np.array(all_errors, dtype='float64'), np.array(all_n_cams))
    
    project_dir = config['project_dir']
    poses_3d_dir = config['directory']['poses_3d']
    labels_filepaths = sorted(glob(os.path.join(project_dir, poses_3d_dir, '*.csv')))
    bodyparts = config['bodyparts']

    all_points, all_scores, all_errors, all_n_cams = load_pose_3d_with_ncams(labels_filepaths, bodyparts)

    errors, n_cams = all_errors[0], all_n_cams[0]

    filtered_errors = np.where(n_cams > 1, errors, np.nan)
    df_errors = pd.DataFrame(filtered_errors.T, columns=bodyparts)
    df_melted = df_errors.melt(var_name='BodyParts', value_name='Error(px)')

    # Remove NaN values and errors above 100
    df_melted = df_melted.dropna()
    df_melted = df_melted[df_melted['Error(px)'] <= 100]

    plt.figure(figsize=(15, 8))
    sns.boxplot(x='BodyParts', y='Error(px)', data=df_melted, fliersize=2, showfliers=False)
    sns.stripplot(x='BodyParts', y='Error(px)', data=df_melted, color='grey', size=0.2, jitter=True)
    plt.title(f'Reprojection Errors for Each Body Part (Frames={errors.shape[1]})')
    plt.xticks(rotation=45)
    y_ticks = np.arange(0, df_melted['Error(px)'].max()+1, step=5)
    plt.yticks(y_ticks)
    plt.tight_layout()
    plt.show()
