import os
import numpy as np
from glob import glob

from marmopose.utils.common import get_camera_group, get_cam_name
from marmopose.utils.io import load_all_poses_2d, save_pose_3d, load_coordinates, load_bodypart_constraints

from typing import List, Tuple, Dict, Any


def project_vector(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Project vector u onto vector v.

    Args:
        u: The vector to be projected.
        v: The vector onto which u is projected.

    Returns:
        The projection of u onto v.
    """
    return u * np.dot(v, u) / np.dot(u, u)


def orthogonalize_vector(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Orthogonalize vector u with respect to vector v.

    Args:
        u: The vector to be orthogonalized.
        v: The vector with respect to which u is orthogonalized.

    Returns:
        The orthogonalized version of u with respect to v.
    """
    return u - project_vector(v, u)


def calculate_3d_axes(config: dict, camera_group: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the 3D axes using triangulation and rotate the coordinate frame to match the axes.

    Args:
        config: Configuration dictionary.
        camera_group: Camera group instance.

    Returns:
        Adjusted 3D axes, rotation matrix, and center of the axes.
    """
    axes_3d_adj, rotation_matrix, center = np.zeros((3, 3)), np.eye(3), np.zeros(3)

    if config['triangulation']['axes']:
        coordinates_path = os.path.join(config['project_dir'], config['directory']['calibration'], 'coordinates.toml')
        axes = load_coordinates(coordinates_path, 'axes')
        offset = np.array(axes['offset'])
        axes_2d = np.array([axes[cam_name] for cam_name in camera_group.get_names()])
        axes_3d = camera_group.triangulate(axes_2d, show_progress=False) - offset

        x_direction = axes_3d[1] - axes_3d[0]
        y_direction = orthogonalize_vector(axes_3d[2] - axes_3d[0], x_direction)

        rotation_matrix = np.array([x_direction, y_direction, np.cross(x_direction, y_direction)])
        rotation_matrix /= np.linalg.norm(rotation_matrix, axis=1)[:, None]

        axes_3d_adj = axes_3d @ rotation_matrix.T
        center = axes_3d_adj[0]
        axes_3d_adj = axes_3d_adj - center # Not in-place

    return axes_3d_adj, rotation_matrix, center


def transform_coordinates(points_3d: np.ndarray, rotation_matrix: np.ndarray, center: np.ndarray) -> np.ndarray:
    """
    Transform the coordinates of 3D points by applying the rotation matrix and adjusting the center.

    Args:
        points_3d: Original 3D points.
        rotation_matrix: Matrix used for rotating the points.
        center: New center for the points.

    Returns:
        Adjusted 3D points.
    """
    points_3d_adj = (points_3d @ rotation_matrix.T) - center
    return points_3d_adj


def triangulate_with_optimization(config: Dict[str, Any], 
                                  bodyparts: List[str], 
                                  points_2d: np.ndarray, 
                                  scores_2d: np.ndarray, 
                                  camera_group: Any, 
                                  verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Handles the case of optimization by triangulating the initial 3D points and optimizing them 
    based on the 2D points, the score, and some constraints.

    Args:
        config: Configuration parameters for triangulation and optimization.
        bodyparts: List of body parts to be considered.
        points_2d: 2D points data with shape of (n_cams, n_frames, n_bodyparts, [x,y] coordinates).
        scores_2d: 2D score data with shape of (n_cams, n_frames, n_bodyparts).
        camera_group: The CameraGroup instance used for triangulation and optimization.
        verbose: If True, the function will display detailed messages about its progress.
        
    Returns:
        points_3d: Optimized 3D points data.
        scores_3d: 3D scores.
        errors: Reprojection errors.
        valid_cams: Number of valid cameras for each point.
    """
    constraints = load_bodypart_constraints(config, bodyparts, 'constraints')
    constraints_weak = load_bodypart_constraints(config, bodyparts, 'constraints_weak')

    n_cams, n_frames, n_bodyparts, _ = points_2d.shape

    points_2d_flat = points_2d.reshape(n_cams, n_frames*n_bodyparts, 2)
    points_3d_init = camera_group.triangulate(points_2d_flat, show_progress=verbose)
    points_3d_init = points_3d_init.reshape((n_frames, n_bodyparts, 3))

    if verbose: print('Optimizing...')
    points_3d = camera_group.optim_points(points_2d, 
                                    points_3d_init,
                                    scores=scores_2d,
                                    constraints=constraints,
                                    constraints_weak=constraints_weak,
                                    scale_smooth=config['triangulation']['scale_smooth'],
                                    scale_length=config['triangulation']['scale_length'],
                                    scale_length_weak=config['triangulation']['scale_length_weak'],
                                    reproj_error_threshold=config['triangulation']['reproj_error_threshold'],
                                    n_deriv_smooth=config['triangulation']['n_deriv_smooth'],
                                    verbose=verbose)
    
    points_3d_flat = points_3d.reshape(-1, 3)

    errors = camera_group.reprojection_error(points_3d_flat, points_2d_flat, mean=True)
    errors = errors.reshape(n_frames, n_bodyparts)

    good_points = ~np.isnan(points_2d[:, :, :, 0])
    valid_cams = np.sum(good_points, axis=0).astype('float')
    scores_2d[~good_points] = 2
    scores_3d = np.min(scores_2d, axis=0)
    scores_3d[valid_cams < 1] = np.nan
    errors[valid_cams < 1] = np.nan

    return points_3d, scores_3d, errors, valid_cams


def triangulate_without_optimization(points_2d: np.ndarray, 
                                     scores_2d: np.ndarray, 
                                     camera_group: Any,
                                     verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Handles the case of non-optimization by directly triangulating the 3D points.

    Args:
        points_2d: 2D points data with shape of (n_cams, n_frames, n_bodyparts, [x,y] coordinates).
        scores_2d: 2D score data with shape of (n_cams, n_frames, n_bodyparts).
        camera_group: The CameraGroup instance used for triangulation.
        verbose: If True, the function will display detailed messages about its progress.

    Returns:
        points_3d: Triangulated 3D points data.
        scores_3d: 3D scores.
        errors: Reprojection errors.
        valid_cams: Number of valid cameras for each point.
    """
    n_cams, n_frames, n_bodyparts, _ = points_2d.shape

    points_2d_flat = points_2d.reshape(n_cams, n_frames*n_bodyparts, 2)
    if verbose: print('Triangulating...')
    points_3d_flat = camera_group.triangulate(points_2d_flat, show_progress=verbose)
    points_3d = points_3d_flat.reshape((n_frames, n_bodyparts, 3))

    
    errors = camera_group.reprojection_error(points_3d_flat, points_2d_flat, mean=True)
    errors = errors.reshape(n_frames, n_bodyparts)

    good_points = ~np.isnan(points_2d[:, :, :, 0])
    valid_cams = np.sum(good_points, axis=0).astype('float')
    scores_2d[~good_points] = 2
    scores_3d = np.min(scores_2d, axis=0)
    scores_3d[valid_cams < 2] = np.nan
    errors[valid_cams < 2] = np.nan
    valid_cams[valid_cams < 2] = np.nan

    return points_3d, scores_3d, errors, valid_cams


def triangulate(config: Dict[str, Any], filtered: bool = False, verbose: bool = False) -> None:
    """
    Generate 3d coordinates for each track based on 2d coordinates and camera parameters.

    Args:
        config: Configuration dictionary.
        filtered: If True, load pose paths that have been filtered.
        verbose: If True, the function will display detailed messages about its progress.
    
    Returns:
        The result is written to a CSV file.
    """
    project_dir = config['project_dir']
    poses_2d_dir = config['directory']['poses_2d']
    poses_2d_filtered_dir = config['directory']['poses_2d_filtered']

    poses_dir = os.path.join(project_dir, poses_2d_filtered_dir) if filtered else os.path.join(project_dir, poses_2d_dir)
    pose_file_paths = sorted(glob(os.path.join(poses_dir, '*.h5')))

    all_cam_points_scores, metadata = load_all_poses_2d(pose_file_paths)

    camera_names = [get_cam_name(path, config['cam_regex']) for path in pose_file_paths]
    camera_group = get_camera_group(config, camera_names)

    all_points = all_cam_points_scores[..., :2] # (n_cams, n_tracks, n_frames, n_bodyparts, 2)
    all_scores = all_cam_points_scores[..., 2] # (n_cams, n_tracks, n_frames, n_bodyparts)

    # Mark points below score threshold as NaN
    invalid = all_scores < config['triangulation']['score_threshold']
    all_points[invalid] = np.nan

    _, n_tracks, _, _, _ = all_points.shape
    bodyparts = metadata['bodyparts']

    axes_3d, rotation_matrix, center = calculate_3d_axes(config, camera_group)
   
    for track_idx in range(n_tracks):
        track_name = metadata["tracks"][track_idx]
        print(f'*************** {track_name} ***************')
        
        if config['triangulation']['optim']:
            points_3d, scores_3d, errors, valid_cams = triangulate_with_optimization(config, bodyparts, all_points[:, track_idx], all_scores[:, track_idx], camera_group, verbose)
        else:
            points_3d, scores_3d, errors, valid_cams = triangulate_without_optimization(all_points[:, track_idx], all_scores[:, track_idx], camera_group, verbose)

        points_3d = transform_coordinates(points_3d, rotation_matrix, center)

        output_dir = os.path.join(config['project_dir'], config['directory']['poses_3d'])
        os.makedirs(output_dir, exist_ok=True)
        output_filepath = os.path.join(output_dir, f'{track_name}.csv')
        save_pose_3d(bodyparts, points_3d, errors, valid_cams, scores_3d, rotation_matrix, axes_3d, output_filepath)


