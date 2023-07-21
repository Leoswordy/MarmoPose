import os
import toml
import numpy as np
import pandas as pd

from typing import List, Tuple, Dict, Any


def save_pose_2d(all_points_scores: np.ndarray, metadata: Dict[str, Any], output_path: str) -> None:
    """
    Write points, scores and metadata in HDF file for all tracks.

    Args:
        all_points_scores : Array containing track points and scores (n_tracks, n_frames, n_bodyparts, [x, y, scores])
        metadata: Metadata for constructing the DataFrame
        output_path: Output file path
    """
    scorer = metadata['scorer']
    tracks = metadata['tracks']
    bodyparts = metadata['bodyparts']
    index = metadata['index']

    columns = pd.MultiIndex.from_product([[scorer], tracks, bodyparts, ['x', 'y', 'likelihood']], 
                                         names=['scorer', 'tracks', 'bodyparts', 'points_scores'])
    data_out = pd.DataFrame(index=index, columns=columns, dtype=float)
    for idx, points_scores in enumerate(all_points_scores):
        points = points_scores[:, :, :2]
        scores = points_scores[:, :, 2]

        data_out.loc[:, (scorer, tracks[idx], bodyparts, 'x')] = points[:, :, 0]
        data_out.loc[:, (scorer, tracks[idx], bodyparts, 'y')] = points[:, :, 1]
        data_out.loc[:, (scorer, tracks[idx], bodyparts, 'likelihood')] = scores

    data_out.to_hdf(output_path, key='poses', mode='w')


def load_pose_2d(file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load 2D poses from HDF file for one camera.

    Args:
        file_path: Path of the HDF file to load data from.

    Returns:
        Tuple containing 2D pose data and metadata.
            - all_points_scores shape of (n_tracks, n_frames, n_bodyparts, [x, y, scores])
    """
    data_frame = pd.read_hdf(file_path)
    scorer = data_frame.columns.levels[0][0]
    pose_data = data_frame.loc[:, scorer]
    tracks = pose_data.columns.levels[0]

    all_points_scores = []
    for track_name in tracks:
        track = pose_data.loc[:, track_name]
        
        bodyparts = list(track.columns.get_level_values(0).unique())
        n_frames = len(track)
        n_bodyparts = len(bodyparts)

        pose = np.array(track).reshape(n_frames, n_bodyparts, 3)
        all_points_scores.append(pose)

    metadata = {
        'scorer': scorer,
        'tracks': tracks, 
        'bodyparts': bodyparts,
        'index': pose_data.index
    }

    return np.array(all_points_scores), metadata


def load_all_poses_2d(file_paths: List[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Loads all pose data from given pose file paths.

    Args:
        pose_file_paths: A list of paths to pose files.

    Returns:
        A tuple containing a numpy array of poses and metadata.
            - all_points_scores shape of (n_cams, n_tracks, n_frames, n_bodyparts, [x, y, scores])
    """
    all_cam_points_scores = []
    for path in file_paths:
        all_points_scores, metadata = load_pose_2d(path)
        all_cam_points_scores.append(all_points_scores)
    return np.array(all_cam_points_scores), metadata


def save_pose_3d(bodyparts: List[str], 
                 points_3d: np.ndarray, 
                 errors: np.ndarray, 
                 valid_cams: np.ndarray, 
                 scores_3d: np.ndarray, 
                 rotation_matrix: np.ndarray, 
                 axes_3d: np.ndarray,
                 output_filepath: str) -> None:
    """
    Save 3D pose data for every track.

    Args:
        bodyparts: List of body part names.
        points_3d: 3D points data with shape of (n_frames, n_bodyparts, [x, y, z] coordinates).
        errors: Reprojection errors.
        valid_cams: Number of valid cameras for each point.
        scores_3d: 3D scores (n_frames, n_bodyparts).
        rotation_matrix: Transformation matrix.
        axes_3d: Axes for the 3D data.
        output_filepath: The path of the output 3D pose file.
    """
    data_dict = {}

    for bp_idx, bp in enumerate(bodyparts):
        for ax_idx, axis in enumerate(['x', 'y', 'z']):
            data_dict[f'{bp}_{axis}'] = points_3d[:, bp_idx, ax_idx]
        data_dict[f'{bp}_error'] = errors[:, bp_idx]
        data_dict[f'{bp}_ncams'] = valid_cams[:, bp_idx]
        data_dict[f'{bp}_score'] = scores_3d[:, bp_idx]

    for i in range(3):
        for j in range(3):
            data_dict[f'rotation_{i}{j}'] = rotation_matrix[i, j]
            data_dict[f'axes_{i}{j}'] = axes_3d[i, j]

    data_out = pd.DataFrame(data_dict)
    data_out.to_csv(output_filepath, index=False)


def load_pose_3d(filepaths: List[str], 
                 bodyparts: List[str]) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Load 3D pose data from csv files for all tracks.

    Args:
        filepaths: List of csv file paths.
        bodyparts: List of body parts.

    Returns:
        Loaded 3D pose data, score data, error data, rotation matrix and 3d axes.
        3D pose data with shape of (n_tracks, n_bodyparts, n_frames, [x, y, z] coordinates)
        score and error data with shape of (n_tracks, n_bodyparts, n_frames)
    """
    all_points, all_scores, all_errors,  = [], [], []
    rotation_matrix, axes_3d = np.zeros((3,3)), np.zeros((3,3))

    for file in filepaths:
        data = pd.read_csv(file)
        points, scores, errors = [], [], []

        for bp in bodyparts:
            points.append(data.loc[:, (bp+'_x', bp+'_y', bp+'_z')].values)
            scores.append(data.loc[:, bp+'_score'].values)
            errors.append(data.loc[:, bp+'_error'].values)

        for i in range(3):
            for j in range(3):
                rotation_matrix[i, j] = data.loc[0, 'rotation_{}{}'.format(i, j)]
                axes_3d[i, j] = data.loc[0, 'axes_{}{}'.format(i, j)]

        all_points.append(points)
        all_scores.append(scores)
        all_errors.append(errors)

    return (np.array(all_points, dtype='float64'), np.array(all_scores, dtype='float64'), 
            np.array(all_errors, dtype='float64'), rotation_matrix, axes_3d)


def save_coordinates(output_filepath: str, key: str, value: Any) -> None:
    """
    Save a key-value pair to a TOML file. If the key already exists in the file, its value will be updated.

    Args:
        output_filepath: Path to the TOML file.
        key: The key to save to the file.
        value: The value to save to the file.
    """
    data_dict = toml.load(output_filepath) if os.path.exists(output_filepath) else {}
    data_dict[key] = value

    with open(output_filepath, 'w') as f:
        toml.dump(data_dict, f)


def load_coordinates(coordinates_filepath: str, key: str) -> Any:
    """
    Load a value from a TOML file given a key.

    Args:
        coordinates_filepath: Path to the TOML file.
        key: The key corresponding to the value to be loaded from the file.

    Returns:
        The value corresponding to the provided key in the TOML file.
    """
    data_dict = toml.load(coordinates_filepath)
    return data_dict[key]


def load_bodypart_constraints(config: Dict[str, Any], bodyparts: List[str], key: str = 'constraints') -> List[List[int]]:
    """
    Load constraints for bodypart indices from the configuration.

    Args:
        config: Configuration dictionary.
        bodyparts: List of bodyparts.
        key: The key in the configuration dictionary for constraints. Defaults to 'constraints'.

    Returns:
         List of constraints represented as pairs of bodypart indices.

    Raises:
        AssertionError: If a bodypart from constraints is not found in the list of bodyparts.
    """
    constraints_names = config.get('triangulation', {}).get(key, [])
    bodypart_indices = {bp_name: idx for idx, bp_name in enumerate(bodyparts)}
    constraints = []

    for bodypart_a, bodypart_b in constraints_names:
        assert bodypart_a in bodypart_indices, f'Bodypart {bodypart_a} from constraints not found in list of bodyparts'
        assert bodypart_b in bodypart_indices, f'Bodypart {bodypart_b} from constraints not found in list of bodyparts'
        constraints.append([bodypart_indices[bodypart_a], bodypart_indices[bodypart_b]])

    return constraints