import numpy as np
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict, Any

from marmopose.utils.data_io import save_points_2d_h5, load_points_2d_h5, load_points_3d_h5
from marmopose.calibration.cameras import CameraGroup


def get_reprojected_points_and_error(config: Dict[str, Any], 
                                     points_3d_source: str = 'optimized', 
                                     points_2d_source: str = 'original', 
                                     verbose: bool = True) -> None:
    """
    Function to calculate reprojected 2D points and their corresponding errors.
    
    Args:
        config: Configuration dictionary containing directory and animal information.
        points_3d_source (optional): Source of the 3D points, either 'original' or 'optimized'. Defaults to 'optimized'.
        points_2d_source (optional): Source of the 2D points, either 'original' or 'filtered'. Defaults to 'original'.
        verbose (optional): Whether to print logs. Defaults to True.
    """
    assert points_3d_source in ['original', 'optimized'], f'Invalid points_3d_source, must be one of: original, optimized'
    assert points_2d_source in ['original', 'filtered'], f'Invalid points_2d_source, must be one of: original, filtered'

    project_dir = Path(config['directory']['project'])
    calibration_path = project_dir / config['directory']['calibration'] / 'camera_params.json'
    points_3d_path = project_dir / config['directory']['points_3d'] / f'{points_3d_source}.h5'
    points_2d_path = project_dir / config['directory']['points_2d'] / f'{points_2d_source}.h5'

    reprojected_points_2d_path = project_dir / config['directory']['points_2d'] / 'reprojected.h5'
    reprojected_points_2d_path.parent.mkdir(parents=True, exist_ok=True)

    all_points_3d = load_points_3d_h5(points_3d_path, verbose=verbose) # (n_tracks, n_frames, n_bodyparts, (x, y, z))
    all_points_with_score_2d = load_points_2d_h5(points_2d_path, verbose=verbose) # (n_cams, n_tracks, n_frames, n_bodyparts, (x, y, score))
    n_cams, n_tracks, n_frames, n_bodyparts, _ = all_points_with_score_2d.shape

    camera_group = CameraGroup.load_from_json(calibration_path)

    all_points_3d_flat = all_points_3d.reshape(-1, 3)
    all_points_2d_reprojected_flat = camera_group.reproject(all_points_3d_flat)

    all_points_2d_reprojected = all_points_2d_reprojected_flat.reshape(n_cams, n_tracks, n_frames, n_bodyparts, 2)
    # Reprojected points have no score, so we need to add a dummy score dimension
    all_scores_2d_reprojected = np.zeros((n_cams, n_tracks, n_frames, n_bodyparts, 1))
    all_points_with_score_reprojected_2d = np.concatenate((all_points_2d_reprojected, all_scores_2d_reprojected), axis=4)

    error_mean = get_mean_reprojection_error(all_points_2d_reprojected, all_points_with_score_2d)
    if verbose:
        print('Average reprojection error:')
        for bp, errors in zip(config['animal']['bodyparts'], error_mean):
            print(f'{bp}: {errors:.2f}')

    for (cam, points_with_score_reprojected_2d) in zip(camera_group.cameras, all_points_with_score_reprojected_2d):
        save_points_2d_h5(points=points_with_score_reprojected_2d,
                          name=cam.get_name(), 
                          file_path=reprojected_points_2d_path, 
                          verbose=verbose)


def get_mean_reprojection_error(all_points_2d_reprojected: np.ndarray, 
                                all_points_with_score_2d: np.ndarray) -> np.ndarray:
    """
    Calculate the mean reprojection error for 2D points.
    
    Args:
        all_points_2d_reprojected: Array of shape (n_cams, n_tracks, n_frames, n_bodyparts, 2) containing reprojected 2D points.
        all_points_with_score_2d: Array of shape (n_cams, n_tracks, n_frames, n_bodyparts, 3) containing 2D points and their scores.
        
    Returns:
        Mean reprojection error for each body part.
    """
    all_errors = np.linalg.norm(all_points_2d_reprojected - all_points_with_score_2d[..., :2], axis=4) # (n_cams, n_tracks, n_frames, n_bodyparts)
    all_scores = all_points_with_score_2d[..., 2]

    valid_mask = all_scores > 0.2

    all_errors_masked = np.where(valid_mask, all_errors, np.nan)  # Replace invalid entries with nan
    
    all_errors_masked_mean = np.nanmean(all_errors_masked, axis=(0, 1, 2))  # (n_bodyparts, )

    return all_errors_masked_mean