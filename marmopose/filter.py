import os
import numpy as np
from glob import glob
from scipy import signal

from marmopose.utils.io import load_pose_2d, save_pose_2d

def interpolate_data(values: np.ndarray) -> np.ndarray:
    """
    Interpolates data to fill NaN values.

    Args:
        values: The data to be interpolated.

    Returns:
        The interpolated data.
    """
    nans = np.isnan(values)
    idx = lambda z: np.nonzero(z)[0]
    out = np.copy(values)
    out[nans] = np.interp(idx(nans), idx(~nans), values[~nans]) if not np.isnan(values).all() else 0
    return out


def medfilt_data(values: np.ndarray, size: int = 15) -> np.ndarray:
    """
    Applies a median filter to the data.

    Args:
        values: The data to be filtered.
        size: The size of the median filter. Defaults to 15.

    Returns:
        The filtered data.
    """
    padsize = size + 5
    vpad = np.pad(values, (padsize, padsize), mode='reflect')
    vpadf = signal.medfilt(vpad, kernel_size=size)
    return vpadf[padsize:-padsize]


def filter_2d(config, verbose: bool = True):
    project_dir = config['project_dir']
    poses_2d_dir = config['directory']['poses_2d']
    poses_2d_filtered_dir = config['directory']['poses_2d_filtered']

    labels_list = sorted(glob(os.path.join(project_dir, poses_2d_dir, '*.h5')))
    output_dir = os.path.join(project_dir, poses_2d_filtered_dir)
    os.makedirs(output_dir, exist_ok=True)

    for labels_filename in labels_list:
        basename = os.path.basename(labels_filename)
        output_filepath = os.path.join(project_dir, poses_2d_filtered_dir, basename)
        
        all_points_scores, metadata = load_pose_2d(labels_filename)
        all_points, all_scores = all_points_scores[..., :2], all_points_scores[..., 2]
        all_points_interp = np.apply_along_axis(interpolate_data, 1, all_points)
        all_points_filtered = np.apply_along_axis(medfilt_data, 1, all_points_interp, size=3)

        save_pose_2d(np.concatenate((all_points_filtered, all_scores[:, :, :, None]), axis=-1), metadata, output_filepath)
    if verbose: print(f'Filtered 2D poses stored in: {output_dir}')