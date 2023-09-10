import numpy as np
from scipy.signal import medfilt
from scipy.interpolate import splev, splrep


def filter_data(points_with_scores: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    """
    Filter the input data based on a given threshold and returns the filtered data.
    
    Args:
        points_with_scores: Input data with shape (n_tracks, n_frames, n_bodyparts, 3).
        threshold: Threshold value for filtering the input data. Default value is 0.2.
    
    Returns:
        Filtered data with shape (n_tracks, n_frames, n_bodyparts, 3).
    """
    points = points_with_scores[:, :, :, :2]
    scores = points_with_scores[:, :, :, 2]

    # TODO: Optimize filtering
    points = np.apply_along_axis(interpolate_data, axis=1, arr=points)

    points[scores < threshold] = np.nan
    
    filtered_data = np.concatenate([points, scores[..., None]], axis=-1)
    
    return filtered_data


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
    vpadf = medfilt(vpad, kernel_size=size)
    return vpadf[padsize:-padsize]


# def interpolate(data: np.ndarray) -> np.ndarray:
#     """
#     Interpolates and apply a median filter to a 1D array of coordinates.

#     Args:
#         data: A 1D array of coordinate values.
        
#     Returns:
#         The interpolated coordinates.
#     """
#     indices = np.arange(len(data))
#     valid_mask = ~np.isnan(data)
#     valid_vals = data[valid_mask]
#     valid_frames = indices[valid_mask]

#     if len(valid_frames) > 0:
#         spline = splrep(valid_frames, valid_vals, k=1)
#         data = splev(indices, spline)

#     return data


# def filter(data: np.ndarray, size: int = 5) -> np.ndarray:
#     """
#     Apply a median filter to a 1D array of coordinates.

#     Args:
#         data: A 1D array of coordinate values.
        
#     Returns:
#         The filtered coordinates.
#     """
#     # TODO: Maybe need padding to avoid edge effects
#     return medfilt(data, kernel_size=size)