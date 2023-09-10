import numpy as np
from scipy import optimize
from scipy.sparse import dok_matrix
from typing import Dict, Any, List, Tuple

from marmopose.processing.filter import interpolate_data
from marmopose.calibration.cameras import CameraGroup
from marmopose.utils.helpers import Timer


def optimize_coordinates(config: Dict[str, Any],
                         camera_group: CameraGroup,
                         points_3d: np.ndarray,
                         points_with_score_2d: np.ndarray,
                         start_frame: int = 0,
                         verbose: bool = False) -> np.ndarray:
    """
    Optimize the 3D points by minimizing the reprojection error, smoothness error, limb length error.

    Args:
        config: Configuration dictionary.
        camera_group: The camera group correponsding to the points.
        points_3d: 3D points with shape (n_frames, n_bodyparts, 3), final channel (x, y, z)
        points_with_score_2d: 2D points with shape (n_cams, n_frames, n_bodyparts, 3), final channel (x, y, score)
        start_frame: Index of the first frame to optimize.
        verbose: If True, the function will print detailed information.

    Returns:
        Optimized 3D points with shape (n_frames, n_bodyparts, 3).

    Example constraints:
        [((0, 2), 30.0), ((0, 1), 30.0)], which means the length of bodypart 0-1 should be approximately 30 mm.
    """
    timer = Timer().start()
    n_deriv_smooth = config['optimization']['n_deriv_smooth']
    scale_smooth = config['optimization']['scale_smooth']
    scale_length = config['optimization']['scale_length']
    scale_length_weak = config['optimization']['scale_length_weak']

    bodypart_dist = parse_constraints(config, 'bodypart_distance')
    bodypart_dist_weak = parse_constraints(config, 'bodypart_distance_weak')
    
    points_3d_interp = np.apply_along_axis(interpolate_data, 0, points_3d)
    points_3d_original, points_3d_unprocessed = points_3d_interp[:start_frame], points_3d_interp[start_frame:]
    points_with_score_2d_unprocessed = points_with_score_2d[:, start_frame:]
    initial_params = points_3d_unprocessed.ravel()
    if verbose: print(f'Optimizing {points_3d_unprocessed.shape[0]} frames, {initial_params.shape[0]} parameters in total')
    jac_sparsity = get_jac_sparsity(points_with_score_2d_unprocessed[..., :2], n_deriv_smooth, bodypart_dist, bodypart_dist_weak, verbose)

    timer.record('init')
    result = optimize.least_squares(fun=compute_residuals, 
                                    x0=initial_params, 
                                    method='trf',
                                    loss='linear',
                                    ftol = 1e-2,
                                    max_nfev = 3,
                                    jac_sparsity=jac_sparsity,
                                    verbose=2*verbose, 
                                    args=(camera_group,
                                          points_with_score_2d_unprocessed, 
                                          n_deriv_smooth,
                                          scale_smooth, 
                                          scale_length,
                                          scale_length_weak,
                                          bodypart_dist,
                                          bodypart_dist_weak))
    points_3d_optimized = result.x.reshape(points_3d_unprocessed.shape)
    residuals = result.fun
    if verbose: print(f'Optimization finished mean residual: {np.mean(residuals):.2f}')
    timer.record('optimize')
    timer.show()

    points_3d_result = np.vstack((points_3d_original, points_3d_optimized))
    return points_3d_result
    

def get_jac_sparsity(points_2d: np.ndarray, n_deriv_smooth: int,
                     bodypart_dist: List[Tuple[Tuple[int, int], float]], 
                     bodypart_dist_weak: List[Tuple[Tuple[int, int], float]], 
                     verbose: bool) -> dok_matrix:
    """
    Calculate Jacobian Sparsity Pattern.

    Args:
        points_2d: 2D points as input.
        n_deriv_smooth: Number of derivatives to smooth.
        bodypart_dist: Strong constraints on body parts.
        bodypart_dist_weak: Weak constraints on body parts.
        verbose: Verbose output flag.

    Returns:
        Sparse Jacobian Matrix.
    """
    n_cams, n_frames, n_bodyparts, _ = points_2d.shape
    n_constraints, n_constraints_weak = len(bodypart_dist), len(bodypart_dist_weak)
    points_2d_flat = points_2d[..., 0].ravel() # (n_cams * n_frames * n_bodyparts,)
    mask_valid = ~np.isnan(points_2d_flat)

    n_errors_reproj = np.sum(mask_valid)
    n_errors_smooth = (n_frames-n_deriv_smooth) * n_bodyparts
    n_errors_lengths = n_constraints * n_frames
    n_errors_lengths_weak = n_constraints_weak * n_frames
    n_errors = n_errors_reproj + n_errors_smooth + n_errors_lengths + n_errors_lengths_weak
    if verbose: print(f'Optimizing {n_errors_reproj} reprojection errors, {n_errors_smooth} smoothness errors, {n_errors_lengths} limb length errors, {n_errors_lengths_weak} weak limb length errors')

    sparse_jac = dok_matrix((n_errors, n_frames*n_bodyparts*3), dtype='int16')

    # Setting the sparsity pattern for reprojection errors
    indices_params = np.tile(np.arange(n_frames*n_bodyparts), n_cams)
    indices_params_valid = indices_params[mask_valid]
    indices_reproj = np.arange(n_errors_reproj)
    for k in range(3):
        sparse_jac[indices_reproj, indices_params_valid*3 + k] = 1

    # Setting the sparsity pattern for smoothness constraint
    frames = np.arange(n_frames-n_deriv_smooth)
    for j in range(n_bodyparts):
        for n in range(n_deriv_smooth+1):
            pa = frames*n_bodyparts + j
            pb = (frames+n)*n_bodyparts + j
            for k in range(3):
                sparse_jac[n_errors_reproj + pa, pb*3 + k] = 1
    
    # Setting the sparsity pattern for strong constraints
    start = n_errors_reproj + n_errors_smooth
    point_indices_3d = np.arange(n_frames*n_bodyparts).reshape((n_frames, n_bodyparts))
    frames = np.arange(n_frames)
    all_constraints = bodypart_dist + bodypart_dist_weak
    for cix, ((bp1, bp2), length) in enumerate(all_constraints):
        pa = point_indices_3d[frames, bp1]
        pb = point_indices_3d[frames, bp2]
        for k in range(3):
            sparse_jac[start + cix*n_frames + frames, pa*3 + k] = 1
            sparse_jac[start + cix*n_frames + frames, pb*3 + k] = 1

    return sparse_jac


# TODO: Confirm if it is necessary
# @jit(forceobj=True, parallel=True)
def compute_residuals(points_3d_flat: np.ndarray, *args: Tuple) -> np.ndarray:
    """
    Compute Residuals for Optimization.

    Args:
        points_3d_flat: Flattened 3D points.
        *args: Additional arguments including camera settings, 2D points, etc.

    Returns:
        Residuals.
    """
    camera_group, points_with_score_2d, n_deriv_smooth, scale_smooth, \
        scale_length, scale_length_weak, bodypart_dist, bodypart_dist_weak = args
    
    n_cams, n_frames, n_joints, _ = points_with_score_2d.shape
    points_3d = points_3d_flat.reshape((n_frames, n_joints, 3))

    errors_reproj = reprojection_residual(camera_group, points_3d, points_with_score_2d)
    errors_smooth = smoothness_residual(points_3d, n_deriv_smooth, scale_smooth)
    errors_lengths = bodypart_length_residual(points_3d, bodypart_dist, bodypart_dist_weak, scale_length, scale_length_weak)
    
    residuals = np.hstack((errors_reproj, errors_smooth, errors_lengths))
    return residuals


def reprojection_residual(camera_group: CameraGroup, points_3d: np.ndarray, points_with_score_2d: np.ndarray) -> np.ndarray:
    """
    Calculate Reprojection Residuals.

    Args:
        camera_group: Group of cameras for reprojection.
        points_3d: 3D coordinates of points.
        points_with_score_2d: 2D coordinates of points with scores.

    Returns:
        Reprojection residuals.
    """
    points_2d, scores_2d = points_with_score_2d[..., :2], points_with_score_2d[..., 2]
    n_cams = points_2d.shape[0]
    points_3d_flat = points_3d.reshape(-1, 3)
    points_2d_flat = points_2d.reshape((n_cams, -1, 2))
    errors = camera_group.reprojection_error(points_3d_flat, points_2d_flat)
    # TODO: Maybe not L2 norm, squared L2 norm?
    errors = np.linalg.norm(errors, axis=2) # (n_cams, n_frames*n_bodyparts)

    # TODO: Set proper scores for nan values
    scores_2d[np.isnan(scores_2d)] = 0
    scores_flat = scores_2d.reshape((n_cams, -1))
    errors = errors * scores_flat
    
    # TODO: If the 2D points was interploated, should they be ignored?
    errors_valid = errors[~np.isnan(errors)] # (n_cams * valid n_frames*n_bodyparts,)
    return errors_valid


def smoothness_residual(points_3d: np.ndarray, n_deriv_smooth: int, scale_smooth: float) -> np.ndarray:
    """
    Calculate Smoothness Residuals.

    Args:
        points_3d: 3D coordinates of points.
        n_deriv_smooth: Number of derivatives for smoothing.
        scale_smooth: Scaling factor for smoothness.

    Returns:
        Smoothness residuals.
    """
    diff = np.diff(points_3d, n=n_deriv_smooth, axis=0)
    # TODO: Maybe not L2 norm, squared L2 norm?
    errors = np.linalg.norm(diff, axis=2).ravel() * scale_smooth # (n_frames-n_deriv_smooth * n_bodyparts,)

    return errors


def bodypart_length_residual(points_3d: np.ndarray,
                             bodypart_dist: List[Tuple[Tuple[int, int], float]],
                             bodypart_dist_weak: List[Tuple[Tuple[int, int], float]],
                             scale_length: float, scale_length_weak: float) -> np.ndarray:
    """
    Calculate Body Part Length Residuals.

    Args:
        points_3d : 3D coordinates of points.
        bodypart_dist: Strong constraints for body parts.
        bodypart_dist_weak: Weak constraints for body parts.
        scale_length: Scaling factor for strong constraints.
        scale_length_weak: Scaling factor for weak constraints.

    Returns:
         Length residuals.
    """
    n_frames = points_3d.shape[0]

    errors = np.empty((len(bodypart_dist), n_frames), dtype='float64')
    for cix, ((bp1, bp2), expected_length) in enumerate(bodypart_dist):
        actual_lengths = np.linalg.norm(points_3d[:, bp1] - points_3d[:, bp2], axis=1)
        # TODO: Maybe not L2 norm, squared L2 norm?
        errors[cix] = np.abs(actual_lengths - expected_length)
    errors = errors.ravel() * scale_length # (n_constraints * n_frames,)

    errors_weak = np.empty((len(bodypart_dist_weak), n_frames), dtype='float64')
    for cix, ((bp1, bp2), expected_length) in enumerate(bodypart_dist_weak):
        actual_lengths = np.linalg.norm(points_3d[:, bp1] - points_3d[:, bp2], axis=1)
        errors_weak[cix] = np.abs(actual_lengths - expected_length)
    errors_weak = errors_weak.ravel() * scale_length_weak

    errors = np.hstack((errors, errors_weak))
    return errors


def parse_constraints(config: Dict[str, Any], key: str) -> List[Tuple[Tuple[int, int], float]]:
    """
    Parse Body Part Constraints from Configuration.

    Args:
        config: Configuration dictionary.
        key: The key to look for in the dictionary.

    Returns:
        Parsed constraints.
    """
    bodyparts = config['animal']['bodyparts']
    bodypart_indices = {bp_name: idx for idx, bp_name in enumerate(bodyparts)}
    
    constraint_dict = config['optimization'][key]
    constraint_list = []
    for key, value in constraint_dict.items():
        bp = tuple([bodypart_indices[bp.strip()] for bp in key.split('-')])
        constraint_list.append((bp, value))
    
    return constraint_list