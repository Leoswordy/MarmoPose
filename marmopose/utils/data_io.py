import json
import logging
from pathlib import Path
from typing import Dict

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def load_axes(file_path: str) -> Dict:
    """
    Load a dictionary of axes from a JSON file.

    Args:
        file_path: The path to the JSON file.

    Returns:
        A dictionary of axes.
    """
    with open(file_path, 'r') as f:
        data_dict = json.load(f)
    return data_dict


def save_points_bboxes_2d_h5(points: np.ndarray, bboxes: np.ndarray, name: str, file_path: Path) -> None:
    """
    Saves 2D points for a camera to an HDF5 file.

    Args:
        points: The 2D points to save. Shape of (n_tracks, n_frames, n_bodyparts, 3), final channel (x, y, score).
        bboxes: The bounding boxes of each instance. Shape of (n_tracks, n_frames, 4), final channel (x1, y1, x2, y2).
        name: The name of the camera.
        file_path: The path to the HDF5 file.
    """
    # Store the points and bounding boxes in the same HDF5 file
    with h5py.File(file_path, 'a') as f:
        points_name = f'{name}_points'
        bboxes_name = f'{name}_bboxes'
        if points_name in f:
            del f[points_name]
            logger.info(f'Overwriting existing {points_name} in {file_path}')
        if bboxes_name in f:
            del f[bboxes_name]
            logger.info(f'Overwriting existing {points_name} in {file_path}')

        f.create_dataset(points_name, data=points)
        f.create_dataset(bboxes_name, data=bboxes)
    
    logger.info(f'Saving 2D points and bboxes for {name} in {file_path}')
    

def load_points_bboxes_2d_h5(file_path: Path) -> np.ndarray:
    """
    Load 2D points with scores from an HDF5 file.

    Args:
        file_path: Path to the HDF5 file.

    Returns: 
        (all_points_with_score_2d, all_bboxes)
        - all_points_with_score_2d: Array of 2D points with scores, sorted by camera name.
            - Shape: (n_cameras, n_tracks, n_frames, n_bodyparts, 3)
            - Final channel: (x, y, score)
        - all_bboxes: Array of bounding boxes, sorted by camera name.
            - Shape: (n_cameras, n_tracks, n_frames, 4)
            - Final channel: (x1, y1, x2, y2)
    """
    all_points_with_score_2d = []
    all_bboxes = []
    with h5py.File(file_path, 'r') as f:
        keys = sorted(set([k.split('_')[0] for k in f.keys()]))
        for name in keys:
            points = f[f'{name}_points'][:]
            bboxes = f[f'{name}_bboxes'][:]
            all_points_with_score_2d.append(points)
            all_bboxes.append(bboxes)
    
    all_points_with_score_2d = np.array(all_points_with_score_2d)
    all_bboxes = np.array(all_bboxes)

    logger.info(f'Loaded 2D points and bboxes from {file_path} with order: {keys}')

    return all_points_with_score_2d, all_bboxes


def save_points_3d_h5(points: np.ndarray, name: str, file_path: Path) -> None:
    """
    Saves 3D points for a track to an HDF5 file.

    Args:
        points: The 3D points to save. Shape of (n_frames, n_bodyparts, 3), final channel (x, y, z).
        name: The name of the track.
        file_path: The path to the HDF5 file.
    """
    with h5py.File(file_path, 'a') as f:
        if name in f:
            del f[name]
            logger.info(f'Overwriting existing {name} in {file_path}')
        f.create_dataset(name, data=points)

    logger.info(f'Saving 3D points for {name} in {file_path}')


def load_points_3d_h5(file_path: Path) -> np.ndarray:
    """
    Load 3D points from an HDF5 file.

    Args:
        file_path: Path to the HDF5 file.

    Returns:
        Array of 3D points, sorted by track name.
            - Shape: (n_tracks, n_frames, n_bodyparts, 3)
            - Final channel: (x, y, z)
    """
    all_points_3d = []
    with h5py.File(file_path, 'r') as f:
        keys = sorted(list(f.keys()))
        for name in keys:
            points = f[name][:]
            all_points_3d.append(points)
            
    all_points_3d = np.array(all_points_3d)
    
    logger.info(f'Loaded 3D points from {file_path} with order: {keys}')
    return all_points_3d


def init_appendable_h5(config) -> None:
    """
    Initializes the HDF5 file with extendable datasets for cameras and tracks.

    Args:
        config: The configuration object.
    """
    n_cams = config.calibration['n_cameras']
    n_tracks = config.animal['n_tracks']
    n_bodyparts = len(config.animal['bodyparts'])

    points_2d_path = Path(config.sub_directory['points_2d']) / 'original.h5'
    with h5py.File(points_2d_path, 'w') as f:
        for cam_idx in range(n_cams):
            camera_name = f'cam{cam_idx+1}'
            points_dataset_name = f'{camera_name}_points'
            bboxes_dataset_name = f'{camera_name}_bboxes'

            f.create_dataset(points_dataset_name,
                             shape=(n_tracks, 0, n_bodyparts, 3),
                             maxshape=(n_tracks, None, n_bodyparts, 3),
                             chunks=(n_tracks, 1, n_bodyparts, 3),
                             dtype='float32')
            f.create_dataset(bboxes_dataset_name,
                             shape=(n_tracks, 0, 4),
                             maxshape=(n_tracks, None, 4),
                             chunks=(n_tracks, 1, 4),
                             dtype='float32')

    points_3d_path = Path(config.sub_directory['points_3d']) / 'original.h5'
    with h5py.File(points_3d_path, 'w') as f:
        for track_idx in range(n_tracks):
            track_name = f'track_{track_idx+1}'
            f.create_dataset(track_name,
                             shape=(0, n_bodyparts, 3),
                             maxshape=(None, n_bodyparts, 3),
                             chunks=(1, n_bodyparts, 3),
                             dtype='float32')


def save_data_online_h5(config,
                        points_with_score_2d_batch: np.ndarray,
                        bboxes_batch: np.ndarray,
                        all_points_3d: np.ndarray):
    """
    Appends data for one frame to the HDF5 file.

    Args:
        config: The configuration object.
        points_with_score_2d_batch: The 2D points with scores, shape (n_cameras, n_tracks, n_bodyparts, 3).
        bboxes_batch: The bounding boxes, shape (n_cameras, n_tracks, 4).
        all_points_3d: The 3D points, shape (n_tracks, n_frames=1, n_bodyparts, 3).
    """
    n_cams = config.calibration['n_cameras']
    n_tracks = config.animal['n_tracks']
    n_bodyparts = len(config.animal['bodyparts'])

    assert points_with_score_2d_batch.shape == (n_cams, n_tracks, n_bodyparts, 3), 'Invalid shape for 2D points'
    assert bboxes_batch.shape == (n_cams, n_tracks, 4), 'Invalid shape for bounding boxes'
    assert all_points_3d.shape == (n_tracks, 1, n_bodyparts, 3), 'Invalid shape for 3D points'

    points_2d_path = Path(config.sub_directory['points_2d']) / 'original.h5'
    with h5py.File(points_2d_path, 'a') as f:
        for cam_idx in range(n_cams):
            camera_name = f'cam{cam_idx+1}'
            points_dataset_name = f'{camera_name}_points'
            bboxes_dataset_name = f'{camera_name}_bboxes'

            points_dataset = f[points_dataset_name]
            bboxes_dataset = f[bboxes_dataset_name]

            points = points_with_score_2d_batch[cam_idx]  # shape (n_tracks, n_bodyparts, 3)
            points = points[:, np.newaxis, :, :]  # shape (n_tracks, 1, n_bodyparts, 3)

            bboxes = bboxes_batch[cam_idx]  # shape (n_tracks, 4)
            bboxes = bboxes[:, np.newaxis, :]  # shape (n_tracks, 1, 4)

            new_len = points_dataset.shape[1] + 1
            points_dataset.resize((n_tracks, new_len, n_bodyparts, 3))
            bboxes_dataset.resize((n_tracks, new_len, 4))

            points_dataset[:, -1:, :, :] = points  # shape (n_tracks, 1, n_bodyparts, 3)
            bboxes_dataset[:, -1:, :] = bboxes  # shape (n_tracks, 1, 4)

    points_3d_path = Path(config.sub_directory['points_3d']) / 'original.h5'
    with h5py.File(points_3d_path, 'a') as f:
        for track_idx in range(n_tracks):
            track_name = f'track_{track_idx+1}'
            track_dataset = f[track_name]

            points_3d = all_points_3d[track_idx]  # shape (n_frames=1, n_bodyparts, 3)

            new_len = track_dataset.shape[0] + 1
            track_dataset.resize((new_len, n_bodyparts, 3))

            track_dataset[-1:] = points_3d  # shape (1, n_bodyparts, 3)