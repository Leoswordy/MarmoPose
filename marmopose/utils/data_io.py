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
            logger.info(f'Overwriting existing {bboxes_name} in {file_path}')

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