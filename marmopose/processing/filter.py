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
    # TODO: Optimize filtering
    points = points_with_scores[:, :, :, :2]
    scores = points_with_scores[:, :, :, 2]
    points = np.apply_along_axis(interpolate_data, axis=1, arr=points)

    points[scores < threshold] = np.nan
    
    filtered_data = np.concatenate([points, scores[..., None]], axis=-1)
    
    return filtered_data

    # filtered_data = medfilter(points_with_scores)
    # return filtered_data


def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def medfilter(all_points_with_score, kernel_size=5, offset_threshold=20, score_threshold=0.2):
    n_tracks, n_frames, n_bodyparts, _ = all_points_with_score.shape

    points_full = all_points_with_score[:, :, :, :2]
    scores_full = all_points_with_score[:, :, :, 2]

    points = np.full((n_tracks, n_frames, n_bodyparts, 2), np.nan, dtype='float64')
    scores = np.empty((n_tracks, n_frames, n_bodyparts), dtype='float64')

    for track_idx in range(n_tracks):
        for bp_ix in range(n_bodyparts):
            x = points_full[track_idx, :, bp_ix, 0]
            y = points_full[track_idx, :, bp_ix, 1]
            score = scores_full[track_idx, :, bp_ix]

            xmed = medfilt(x, kernel_size=kernel_size)
            ymed = medfilt(y, kernel_size=kernel_size)

            errx = np.abs(x - xmed)
            erry = np.abs(y - ymed)
            err = np.sqrt(errx**2 + erry**2)

            bad = np.zeros(len(x), dtype='bool')
            bad[err >= offset_threshold] = True
            bad[score < score_threshold] = True
            bad[np.isnan(x)] = True

            Xf = np.array([x,y]).T
            Xf[bad] = np.nan

            Xfi = np.copy(Xf)

            for i in range(Xf.shape[1]):
                vals = Xfi[:, i]
                nans, ix = nan_helper(vals)
                # some data missing, but not too much
                if np.sum(nans) > 0 and np.mean(~nans) > 0.5 and np.sum(~nans) > 5:
                    spline = splrep(ix(~nans), vals[~nans], k=3, s=0)
                    vals[nans]= splev(ix(nans), spline)
                Xfi[:,i] = vals

            points[track_idx, :, bp_ix, 0] = Xfi[:, 0]
            points[track_idx, :, bp_ix, 1] = Xfi[:, 1]
            # dout[scorer, bp, 'interpolated'] = np.isnan(Xf[:, 0])

        # scores = scores_full[:, :, 0]

    # return points, scores
    return np.concatenate([points, scores[..., None]], axis=-1)


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

    # interval = 30
    # for i in range(interval, len(values)-interval):
    #     if np.isnan(values[i-interval:i+interval]).all():
    #         out[i] = np.nan
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