import numpy as np
from pathlib import Path
from typing import Dict, Any

from marmopose.utils.data_io import load_points_2d_h5, save_points_3d_h5
from marmopose.calibration.cameras import CameraGroup
from marmopose.processing.optimization import optimize_coordinates


def triangulate(config: Dict[str, Any], points_2d_source: str = 'filtered', verbose: bool = True) -> None:
    """
    Triangulate 2D points to 3D coordinates using camera calibration parameters.

    Args:
        config: A dictionary containing configuration parameters.
        points_2d_source (optional): The source of 2D points to use for triangulation. Must be one of: 'original', 'filtered'. Defaults to 'filtered'.
        verbose (optional): Whether to print progress messages. Defaults to True.

    Raises:
        AssertionError: If `points_2d_source` is not one of: 'original', 'filtered'.
    """
    assert points_2d_source in ['original', 'filtered'], f'Invalid points_2d_source, must be one of: original, filtered'

    project_dir = Path(config['directory']['project'])
    calibration_path = project_dir / config['directory']['calibration'] / 'camera_params.json'
    points_2d_path = project_dir / config['directory']['points_2d'] / f'{points_2d_source}.h5'
    points_3d_path = project_dir / config['directory']['points_3d'] / 'original.h5'
    points_3d_optimized_path = project_dir / config['directory']['points_3d'] / 'optimized.h5'
    points_3d_path.parent.mkdir(parents=True, exist_ok=True)

    n_tracks = config['animal']['number']

    camera_group = CameraGroup.load_from_json(calibration_path)

    all_points_with_score_2d = load_points_2d_h5(points_2d_path, verbose=verbose)

    for track_idx in range(n_tracks):
        track_name = f'track{track_idx+1}'

        points_with_score_2d = all_points_with_score_2d[:, track_idx] # (n_cams, n_frames, n_bodyparts, (x, y, score)))
        points_3d = reconstruct_3d_coordinates(points_with_score_2d, camera_group, verbose) # (n_frames, n_bodyparts, (x, y, z))
        save_points_3d_h5(points=points_3d, 
                          name=track_name, 
                          file_path=points_3d_path, 
                          verbose=verbose)
        
        if config['optimization']['enable']:
            points_3d_optimized = optimize_coordinates(config, camera_group, points_3d, points_with_score_2d, 0, verbose)
            save_points_3d_h5(points=points_3d_optimized, 
                              name=track_name, 
                              file_path=points_3d_optimized_path, 
                              verbose=verbose)


def reconstruct_3d_coordinates(points_with_score_2d: np.ndarray, camera_group: CameraGroup, verbose: bool = False) -> np.ndarray:
    """
    Reconstructs 3D coordinates from 2D points using triangulation.

    Args:
        points_with_score_2d: Array of shape (n_cams, n_frames, n_bodyparts, 3) containing 2D points and their scores.
        camera_group: CameraGroup object containing camera parameters.
        verbose (optional): Whether to print verbose output. Defaults to False.

    Returns:
        Array of shape (n_frames, n_bodyparts, 3) containing reconstructed 3D coordinates.
    """
    
    n_cams, n_frames, n_bodyparts, _ = points_with_score_2d.shape
    points_2d, scores_2d = points_with_score_2d[..., :2], points_with_score_2d[..., 2]
    points_2d_flat = points_2d.reshape(n_cams, n_frames*n_bodyparts, 2)

    points_3d_flat = camera_group.triangulate(points_2d_flat, undistort=True, verbose=verbose)
    points_3d = points_3d_flat.reshape((n_frames, n_bodyparts, 3))

    return points_3d
