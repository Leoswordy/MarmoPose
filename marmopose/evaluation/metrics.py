import sleap
import numpy as np
from pathlib import Path
from typing import Dict, Any

from marmopose.utils.data_io import save_points_2d_h5, load_points_2d_h5, load_points_3d_h5
from marmopose.calibration.cameras import CameraGroup
from marmopose.processing.triangulation import reconstruct_3d_coordinates


def get_frame_idxs(labels):
    """
    Get indices of frame timestamps that have been labeled. Note that this index starts from 0, while the GUI starts from 1.

    Args: 
        labels: SLEAP labels object.
    """
    frame_idxs_set = set()
    for lf in labels.user_labeled_frames:
        frame_idxs_set.add(lf.frame_idx)
    frame_idxs = sorted(frame_idxs_set)

    return frame_idxs


def triangulate_labeled(camera_group, all_points_with_score_2d, frame_idxs):
    """
    Triangulate a subset of labeled 2D points to 3D coordinates.

    Args:
        camera_group: CameraGroup object containing camera parameters.
        all_points_with_score_2d: Array of shape (n_cams, n_tracks, n_frames, n_bodyparts, 3) containing 2D points and their scores.
        frame_idxs: List of indices of frame timestamps that have been labeled.

    Returns:
        Array of shape (m_tracks, n_labeled_frames, n_bodyparts, 3) containing reconstructed 3D coordinates.
    """
    all_points_3d = []
    for track_idx in range(len(all_points_with_score_2d[1])):
        points_with_score_2d = all_points_with_score_2d[:, track_idx, frame_idxs] # (n_cams, n_labeled_frames, n_bodyparts, (x, y, score)))
        all_points_3d.append(reconstruct_3d_coordinates(points_with_score_2d, camera_group, False))
    
    return np.array(all_points_3d) # (n_tracks, n_labeled_frames, n_bodyparts, (x, y, z))


## =====================Below need to be refactored========================

def evaluate_2d_from_h5(labels_gt_path, pr_path):
    labels_gt = sleap.load_file(str(labels_gt_path))
    all_points_pr_2d = load_points_2d_h5(pr_path) #(n_cams, n_tracks, n_frames, n_bodyparts, 3)

    metrics = []
    for i, video in enumerate(labels_gt.videos):
        frame_idxs = [lf.frame_idx for lf in labels_gt.user_labeled_frames if lf.video == video]

        points_gt_2d = labels_gt.numpy(video=i, return_confidence=True)
        points_gt_2d = np.swapaxes(points_gt_2d, 0, 1)

        points_pr_2d = all_points_pr_2d[i] # (n_tracks, n_frames, n_bodyparts, (x, y, score))

        d = {}
        d['video'] = video.filename
        d.update(compute_id_acc(points_gt_2d[:, :, :], points_pr_2d[:, :, :], frame_idxs))
        d['oks'] = compute_oks(points_gt_2d[:, frame_idxs, :, :2], points_pr_2d[:, frame_idxs, :, :2], stddev=1, scale=50)
        
        metrics.append(d)

    return metrics


def evaluate_2d_from_labels(labels_gt_path, labels_pr_path):
    labels_gt = sleap.load_file(str(labels_gt_path))
    labels_pr = sleap.load_file(str(labels_pr_path))

    metrics = []
    for i, video in enumerate(labels_gt.videos):
        print(video)
        frame_idxs = [lf.frame_idx for lf in labels_gt.user_labeled_frames if lf.video == video]

        points_gt_2d = labels_gt.numpy(video=i, return_confidence=True)
        points_gt_2d = np.swapaxes(points_gt_2d, 0, 1)

        points_pr_2d = labels_pr.numpy(video=i, return_confidence=True)
        points_pr_2d = np.swapaxes(points_pr_2d, 0, 1)

        d = {}
        d['video'] = video.filename
        d.update(compute_id_acc(points_gt_2d[:, :, :], points_pr_2d[:, :, :], frame_idxs))
        d['oks'] = compute_oks(points_gt_2d[:, frame_idxs, :, :2], points_pr_2d[:, frame_idxs, :, :2], stddev=1, scale=50)
        
        metrics.append(d)
    
    return metrics


def compute_id_acc(points_gt_2d, points_pr_2d, frame_idxs):
    n_tracks, n_frames, n_bodyparts, _ = points_gt_2d.shape

    miss, wrong, correct = 0, 0, 0
    miss_list, wrong_list = [], []
    for frame_idx in frame_idxs:
        frame_gt = points_gt_2d[:, frame_idx, :, :2]
        if ((~np.isnan(frame_gt).any(axis=-1)).sum(axis=-1) < 4).all():
            # not gt, no need to evaluate
            continue

        for track_idx in range(n_tracks):
            instance_gt = points_gt_2d[track_idx, frame_idx, :, :2]
            instance_pr = points_pr_2d[track_idx, frame_idx, :, :2]
            if np.isnan(instance_pr).all():
                if np.sum(~np.isnan(instance_gt).any(axis=-1)) > 6:
                    miss_list.append(frame_idx)
                    miss += 1
                continue

            similarity = compute_instance_similarity(frame_gt, instance_pr[np.newaxis, :, :])

            if np.all(np.isnan(similarity)):
                continue
            if np.nanmax(similarity)-similarity[track_idx] < 0.1:
                correct += 1
            else:
                wrong_list.append(frame_idx)
                wrong += 1

    return {
        'miss': miss,
        'wrong': wrong,
        'correct': correct,
        'miss_list': miss_list,
        'wrong_list': wrong_list,
    }


def compute_instance_similarity(points_gt, points_pr, scale=100):
    ref_visible = ~(np.isnan(points_gt).any(axis=-1))
    sum_ref_visible = np.sum(ref_visible, axis=1) # (n_tracks, )
    dists = np.sum((points_gt - points_pr) ** 2, axis=-1)  # (n_tracks, n_bodyparts)
    # similarity = np.nansum(np.exp(-dists/scale**2), axis=-1) / np.sum(ref_visible, axis=1)
    similarity = np.divide(
        np.nansum(np.exp(-dists / scale ** 2), axis=-1),
        sum_ref_visible,
        out=np.full_like(sum_ref_visible, np.nan, dtype=float),
        where=(sum_ref_visible != 0)
    )

    return similarity


def compute_oks_2d(points_gt_2d, points_pr_2d, frame_idxs):
    """
    Args:
        points_gt_2d: (n_cams, n_tracks, n_frames, n_bodyparts, (x, y, score))
        points_pr_2d: (n_cams, n_tracks, n_frames, n_bodyparts, (x, y, score))
    """

    oks = np.array([compute_oks(points_gt_2d[i, :, frame_idxs, :, :2], points_pr_2d[i, :, frame_idxs, :, :2], stddev=1, scale=50) 
                    for i in range(len(points_gt_2d))])
    
    return oks


def remove_outliers(points_pr_2d):
    REMOVE_LIST = [
        [7952, 9194],
        [7952],
        [3111, 9911, 9912, 9913, 17132],
        [471, 472, 473, 474, 475, 16541, 16542, 16543, 16544, 16545, 16546, 16551, 16552, 16999, 17067, 17132, 17174, 17999]
    ]
    for cam_idx in range(len(REMOVE_LIST)):
        for frame_idx in REMOVE_LIST[cam_idx]:
            points_pr_2d[cam_idx, :, frame_idx, :, :] = np.nan


def evaluate_3d_from_labels(labels_gt_path, labels_pr_path, camera_path):
    # Load camera_path
    camera_group = CameraGroup.load_from_json(str(camera_path))

    # Load and compute points_gt_3d
    labels_gt = sleap.load_file(str(labels_gt_path))
    frame_idxs = get_frame_idxs(labels_gt)
    points_gt_2d = np.array([labels_gt.numpy(video=i, return_confidence=True) for i in range(len(labels_gt.videos))])
    points_gt_2d = np.swapaxes(points_gt_2d, 1, 2) # (n_cams, n_tracks, n_frames, n_bodyparts, 3)
    points_gt_3d = triangulate_labeled(camera_group, points_gt_2d, frame_idxs) # (n_tracks, n_labeled_frames, n_bodyparts, (x, y, z))

    # Load and compute points_pr_3d
    labels_pr = sleap.load_file(str(labels_pr_path))
    labels_pr.videos = sorted(labels_pr.videos, key=lambda x: x.filename)
    points_pr_2d = np.array([labels_pr.numpy(video=i, return_confidence=True) for i in range(len(labels_pr.videos))]) # (n_tracks, n_cams, n_frames, n_bodyparts, 3)
    points_pr_2d = np.swapaxes(points_pr_2d, 1, 2) # (n_cams, n_tracks, n_frames, n_bodyparts, 3)
    # remove_outliers(points_pr_2d)
    points_pr_3d = triangulate_labeled(camera_group, points_pr_2d, frame_idxs) # (n_tracks, n_labeled_frames, n_bodyparts, (x, y, z))
    
    # Compute metrics
    metrics = dict()
    metrics_3d = compute_metrics(points_gt_3d, points_pr_3d)
    metrics.update(metrics_3d)

    return metrics


def evaluate_3d_from_h5(labels_gt_path, pr_path, camera_path):
    camera_group = CameraGroup.load_from_json(str(camera_path))

    labels_gt = sleap.load_file(str(labels_gt_path))
    frame_idxs = get_frame_idxs(labels_gt)
    
    points_gt_2d = np.array([labels_gt.numpy(video=i, return_confidence=True) for i in range(len(labels_gt.videos))])
    points_gt_2d = np.swapaxes(points_gt_2d, 1, 2) # (n_cams, n_tracks, n_frames, n_bodyparts, 3)
    points_gt_3d = triangulate_labeled(camera_group, points_gt_2d, frame_idxs) # (n_tracks, n_labeled_frames, n_bodyparts, (x, y, z))

    points_pr_3d = load_points_3d_h5(pr_path) # (n_tracks, n_frames, n_bodyparts, (x, y, z))
    points_pr_3d = points_pr_3d[:, frame_idxs] # (n_tracks, n_labeled_frames, n_bodyparts, (x, y, z))
    
    metrics = dict()
    metrics_3d = compute_metrics(points_gt_3d, points_pr_3d)
    metrics.update(metrics_3d)

    return metrics


def compute_metrics(points_gt_3d, points_pr_3d):
    metrics = dict()

    metrics.update(compute_dist(points_gt_3d, points_pr_3d))
    metrics.update(compute_visibility(points_gt_3d, points_pr_3d))

    oks = compute_oks(points_gt_3d, points_pr_3d, stddev=0.025)
    metrics['oks'] = oks
    metrics['oks.avg'] = np.nanmean(oks, axis=(0, 1))
    metrics.update(compute_generalized_voc_metrics(oks))

    return metrics


def compute_dist(points_gt_3d, points_pr_3d):
    dists = np.linalg.norm(points_gt_3d - points_pr_3d, axis=-1)
    # dists[dists > 100] = np.nan # remove outliers

    results = {
        'dists': dists,
        'dist.median': np.nanmedian(dists, axis=(0, 1)),
        'dist.avg': np.nanmean(dists, axis=(0, 1)),
        'dist.std': np.nanstd(dists, axis=(0, 1))
    }

    return results


def compute_visibility(points_gt_3d, points_pr_3d):
    missing_nodes_gt = np.isnan(points_gt_3d).any(axis=-1)
    missing_nodes_pr = np.isnan(points_pr_3d).any(axis=-1)

    vis_tn = ((missing_nodes_gt) & (missing_nodes_pr)).sum()
    vis_fn = ((~missing_nodes_gt) & (missing_nodes_pr)).sum()
    vis_fp = ((missing_nodes_gt) & (~missing_nodes_pr)).sum()
    vis_tp = ((~missing_nodes_gt) & (~missing_nodes_pr)).sum()

    return {
        "vis.tp": vis_tp,
        "vis.fp": vis_fp,
        "vis.tn": vis_tn,
        "vis.fn": vis_fn,
        "vis.precision": vis_tp / (vis_tp + vis_fp) if (vis_tp + vis_fp) else np.nan,
        "vis.recall": vis_tp / (vis_tp + vis_fn) if (vis_tp + vis_fn) else np.nan,
    }


def compute_oks(points_gt: np.ndarray, points_pr: np.ndarray, stddev: float = 0.025, scale: float = 100) -> np.ndarray:
    n_tracks, n_frames, n_bodyparts, _ = points_gt.shape

    # scale = compute_instance_area(points_gt) # (n_tracks, n_frames)
    if np.isscalar(stddev):
        stddev = np.full(n_bodyparts, stddev)
    if np.isscalar(scale):
        scale = np.full((n_tracks, n_frames), scale**(points_gt.shape[-1]))
    
    distance = np.sum((points_gt - points_pr) ** 2, axis=-1)  # (n_tracks, n_frames, n_bodyparts)

    spread_factor = stddev ** 2
    scale_factor = 2 * ((scale + np.spacing(1)) ** 2)
    normalization_factor = np.reshape(scale_factor, (n_tracks, n_frames, 1)) * np.reshape(spread_factor, (1, 1, n_bodyparts))

    missing_pr = np.any(np.isnan(points_pr), axis=-1) # (n_tracks, n_frames, n_bodyparts)
    distance[missing_pr] = np.inf

    ks = np.exp(-(distance / normalization_factor))  # (n_tracks, n_frames, n_bodyparts)

    missing_gt = np.any(np.isnan(points_gt), axis=-1)  # (n_tracks, n_frames, n_bodyparts)
    ks[missing_gt] = 0

    n_visible_gt = np.sum((~missing_gt), axis=-1) # (n_tracks, n_frames)
    # oks = np.sum(ks, axis=-1) / n_visible_gt # (n_tracks, n_frames)
    oks = np.divide(
        np.nansum(ks, axis=-1),
        n_visible_gt,
        out=np.full_like(n_visible_gt, np.nan, dtype=float),
        where=(n_visible_gt != 0)
    )

    return oks


def compute_generalized_voc_metrics(
    match_scores,
    match_score_thresholds: np.ndarray = np.linspace(0.5, 0.95, 10),  # 0.5:0.05:0.95
    recall_thresholds: np.ndarray = np.linspace(0, 1, 101),  # 0.0:0.01:1.00
    name: str = "oks"
):
    match_scores = sorted(match_scores.flatten(), reverse=True)
    npig = len(match_scores)

    precisions = []
    recalls = []

    for match_score_threshold in match_score_thresholds:
        tp = np.cumsum(match_scores >= match_score_threshold)
        fp = np.cumsum(match_scores < match_score_threshold)

        rc = tp / npig
        pr = tp / (fp + tp + np.spacing(1))

        recall = rc[-1]  # best recall at this OKS threshold

        # Ensure strictly decreasing precisions.
        for i in range(len(pr) - 1, 0, -1):
            if pr[i] > pr[i - 1]:
                pr[i - 1] = pr[i]

        # Find best precision at each recall threshold.
        rc_inds = np.searchsorted(rc, recall_thresholds, side="left")
        precision = np.zeros(rc_inds.shape)
        is_valid_rc_ind = rc_inds < len(pr)
        precision[is_valid_rc_ind] = pr[rc_inds[is_valid_rc_ind]]

        precisions.append(precision)
        recalls.append(recall)

    precisions = np.array(precisions)
    recalls = np.array(recalls)

    AP = precisions.mean(axis=1)  # AP = average precision over fixed set of recall thresholds
    AR = recalls  # AR = max recall given a fixed number of detections per image

    mAP = precisions.mean()  # mAP = mean over all OKS thresholds
    mAR = recalls.mean()  # mAR = mean over all OKS thresholds

    return {
        name + ".match_score_thresholds": match_score_thresholds,
        name + ".recall_thresholds": recall_thresholds,
        name + ".match_scores": match_scores,
        name + ".precisions": precisions,
        name + ".recalls": recalls,
        name + ".AP": AP,
        name + ".AR": AR,
        name + ".mAP": mAP,
        name + ".mAR": mAR,
    }

    
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