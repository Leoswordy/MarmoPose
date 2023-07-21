import os
import numpy as np
from glob import glob
from typing import List, Dict, Any

import sleap
from marmopose.utils.common import Timer
from marmopose.utils.io import save_pose_2d


def predict(config: Dict[str, Any], multi_animal: bool = False, verbose: bool = True) -> None:
    """
    Predict 2D poses using a SLEAP model.

    Args:
        config: Configuration dictionary.
        multi_animal: Indicator whether multiple animals are in the video. Defaults to False.
        verbose: Controls whether to display additional information. Defaulats to True.
    """
    model_dir = config['model_dir']
    project_dir = config['project_dir']
    poses_2d_dir = config['directory']['poses_2d']
    videos_raw_dir = config['directory']['videos_raw']
    video_extension = config['video_extension']
    n_tracks = config['n_tracks']

    video_paths = sorted(glob(os.path.join(project_dir, videos_raw_dir, f'*.{video_extension}')))

    for video_path in video_paths:
        video = sleap.load_video(video_path)
        slp_path = os.path.splitext(video_path)[0] + '.slp'

        if os.path.exists(slp_path):
            if verbose: print(f'Loading labels from: {slp_path}')
            labels = sleap.load_file(slp_path)
        else:
            if verbose: print(f'Predicting labels for: {video_path}')
            labels = predict_labels(video, model_dir, multi_animal, n_tracks, slp_path, verbose)

        output_dir = os.path.join(project_dir, poses_2d_dir)
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f'{basename}.h5')
        export_labels_h5(labels, model_dir, config['bodyparts'], output_path)


def predict_labels(video: sleap.Video, model_dir: str, multi_animal: bool, n_tracks: int, slp_path: str, verbose: bool) -> sleap.Labels:
    """
    Predict labels for a video.

    Args:
        video: Video data.
        model_dir: Model directory path.
        multi_animal: Indicator whether multiple animals are in the video.
        n_tracks: Number of tracks.
        slp_path: SLP file name.
        verbose: Controls whether to display additional information.

    Returns:
        labels: Predicted labels for the video.
    """
    progress_reporting = 'none' if not verbose else 'rich'
    if multi_animal:
        model_paths = [os.path.join(model_dir, 'centroid'), os.path.join(model_dir, 'centered_instance')]
        predictor = sleap.load_model(model_paths, batch_size=8, progress_reporting=progress_reporting)
    else:
        predictor = sleap.load_model(model_dir, batch_size=8, progress_reporting=progress_reporting)

    timer = Timer().start()

    labels = predictor.predict(video)
    timer.record('Predict')

    if multi_animal:
        labels = track(labels, n_tracks)
        timer.record('Track')

    labels.save(slp_path, with_images=False, embed_all_labeled=False)
    timer.record('Save')
    if verbose: timer.show()

    return labels


def track(labels: sleap.Labels, n_tracks: int) -> sleap.Labels:
    """
    Track instances across all of the frames.

    Args:
        labels: Labels for a video.
        n_tracks: Number of instances in every frame.
    
    Return:
        Tracked labels.
    """
    tracker = sleap.nn.tracking.Tracker.make_tracker_by_name(
        # General tracking options
        tracker="flow",
        track_window=5,

        # Matching options
        similarity="instance",
        match="greedy",
        min_new_track_points=1,
        min_match_points=1,

        # Optical flow options (only applies to "flow" tracker)
        img_scale=0.5,
        of_window_size=21,
        of_max_levels=3,

        # Pre-tracking filtering options
        target_instance_count=n_tracks,
        pre_cull_to_target=True,
        pre_cull_iou_threshold=0.8,

        # Post-tracking filtering options
        post_connect_single_breaks=True,
        clean_instance_count=0,
        clean_iou_threshold=None,
    )

    tracked_lfs = []
    for lf in labels:
        lf.instances = tracker.track(lf.instances, img=lf.image)
        tracked_lfs.append(lf)
    tracked_labels = sleap.Labels(tracked_lfs)
    return tracked_labels


def export_labels_h5(labels: sleap.Labels, model_dir: str, bodyparts: List[str], output_path: str) -> None:
    """
    Prepare output directory and export labels.

    Args:
        labels: Labels for the video.
        model_dir: Model directory path.
        bodyparts: The name list of the bodyparts.
        output_path: The path of the output file.
    """
    all_points_scores = labels.numpy(untracked=False, return_confidence=True) #(n_frames, n_tracks, n_bodyparts, 3)
    all_points_scores = np.swapaxes(all_points_scores, 0, 1) #(n_tracks, n_frames, n_bodyparts, 3)

    metadata = {
        'scorer': model_dir,
        'tracks': [f'track{i+1}' for i in range(len(all_points_scores))],
        'bodyparts': bodyparts,
        'index': np.arange(all_points_scores.shape[1])
    }

    save_pose_2d(all_points_scores, metadata, output_path)
