import h5py
import json
import numpy as np
from pathlib import Path
from typing import Dict


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


def save_points_2d_h5(points: np.ndarray, name: str, file_path: Path, verbose: bool = False) -> None:
    """
    Saves 2D points for a camera to an HDF5 file.

    Args:
        points: The 2D points to save. Shape of (n_tracks, n_frames, n_bodyparts, 3), final channel (x, y, score).
        name: The name of the camera.
        file_path: The path to the HDF5 file.
        verbose (optional): Whether to print progress messages. Defaults to False.
    """
    with h5py.File(file_path, 'a') as f:
        if name in f:
            del f[name]
        f.create_dataset(name, data=points)
    if verbose: print(f'Saving 2D points for {name} in {file_path}')
    

def load_points_2d_h5(file_path: Path, verbose: bool = False) -> np.ndarray:
    """
    Load 2D points with scores from an HDF5 file.

    Args:
        file_path: Path to the HDF5 file.
        verbose (optional): Whether to print progress messages. Defaults to False.

    Returns:
        Array of 2D points with scores, sorted by camera name.
            - Shape: (n_cams, n_tracks, n_frames, n_bodyparts, 3)
            - Final channel: (x, y, score)
    """
    all_points_with_score_2d = []
    with h5py.File(file_path, 'r') as f:
        keys = sorted(list(f.keys()))
        for name in keys:
            points = f[name][:]
            all_points_with_score_2d.append(points)
            
    all_points_with_score_2d = np.array(all_points_with_score_2d)
    if verbose: print(f'Loading 2D points from {file_path} with order: {keys}')
    return all_points_with_score_2d


def save_points_3d_h5(points: np.ndarray, name: str, file_path: Path, verbose: bool = False) -> None:
    """
    Saves 3D points for a track to an HDF5 file.

    Args:
        points: The 3D points to save. Shape of (n_frames, n_bodyparts, 3), final channel (x, y, z).
        name: The name of the track.
        file_path: The path to the HDF5 file.
        verbose (optional): Whether to print progress messages. Defaults to False.
    """
    with h5py.File(file_path, 'a') as f:
        if name in f:
            del f[name]
        f.create_dataset(name, data=points)
    if verbose: print(f'Saving 3D points for {name} in {file_path}')


def load_points_3d_h5(file_path: Path, verbose: bool = False) -> np.ndarray:
    """
    Load 3D points from an HDF5 file.

    Args:
        file_path: Path to the HDF5 file.
        verbose (optional): Whether to print progress messages. Defaults to False.

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
    if verbose: print(f'Loading 3D points from {file_path} with order: {keys}')
    return all_points_3d