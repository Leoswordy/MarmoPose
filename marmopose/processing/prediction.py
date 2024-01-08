import numpy as np
from pathlib import Path
from typing import Dict, Any

import sleap
from sleap.nn.tracking import Tracker, MatchedFrameInstances, SimpleCandidateMaker
from sleap.nn.tracker.components import FrameMatches

from marmopose.utils.data_io import save_points_2d_h5
from marmopose.processing.filter import filter_data


def predict(config: Dict[str, Any], batch_size: int = 4, verbose: bool = True) -> None:
    """
    Predicts 2D points for each video in the raw videos directory specified in the configuration file.
    
    Args:
        config: Configuration dictionary containing directory paths and other parameters.
        batch_size (optional): Batch size to use during inference. Default is 4.
        verbose (optional): Whether to print progress messages. Default is True.
    """
    model_dir = config['directory']['model']
    project_dir = Path(config['directory']['project'])
    n_tracks = config['animal']['number']
    videos_raw_dir = project_dir / config['directory']['videos_raw']
    points_2d_path = project_dir / config['directory']['points_2d'] / 'original.h5'
    points_2d_path.parent.mkdir(parents=True, exist_ok=True)
    points_2d_filtered_path = project_dir / config['directory']['points_2d'] / 'filtered.h5'
    points_2d_filtered_path.parent.mkdir(parents=True, exist_ok=True)
    # Find all videos in the raw videos directory
    video_paths = sorted(videos_raw_dir.glob(f"*.{config['video_extension']}"))
    
    # Predict 2D points for each video and save to an HDF5 file
    for video_path in video_paths:
        points_with_score_2d, points_with_score_2d_filtered = predict_points_2d(video_path, n_tracks, model_dir, batch_size, verbose)
        save_points_2d_h5(points=points_with_score_2d,
                          name=video_path.stem, 
                          file_path=points_2d_path, 
                          verbose=verbose)
        
        save_points_2d_h5(points=points_with_score_2d_filtered,
                          name=video_path.stem, 
                          file_path=points_2d_filtered_path, 
                          verbose=verbose)


def predict_points_2d(video_path: Path, n_tracks: int, model_dir: str, batch_size: int = 4 , verbose: bool = True) -> np.ndarray:
    """
    Predicts 2D points for a given video using a trained model.

    Args:
        video_path: Path to the input video.
        model_dir: Path to the directory containing the trained model.
        batch_size (optional): Batch size for prediction. Defaults to 4.
        verbose (optional): Whether to print progress messages. Defaults to True.

    Returns:
        Array of predicted 2D points with confidence scores. 
            - Shape: (n_tracks, n_frames, n_bodyparts, 3)
            - Final channel: (x, y, score)
    """
    
    slp_path = video_path.with_suffix('.slp')
    if slp_path.exists():
        if verbose: print(f'Loading labels from: {slp_path}')
        labels = sleap.load_file(str(slp_path))
    else:
        if verbose: print(f'Predicting labels for: {video_path}')
        video = sleap.load_video(str(video_path))
        predictor = sleap.load_model(model_dir, 
                                     batch_size = batch_size, 
                                     peak_threshold = 0.2,
                                     progress_reporting = 'rich')
        labels = predictor.predict(video)
        labels.save(slp_path, with_images=False, embed_all_labeled=False)
    
    points_with_score_2d = get_arr_from_labels(labels, n_tracks)

    # Tracking
    print(f'Tracking {slp_path}')
    tracker = IDTracker(
        track_window=4,
        similarity_function=instance_similarity,
        candidate_maker=SimpleCandidateMaker(min_points=4),
        target_instance_count=2
    )
    tracked_labels = tracker.track_labels(labels)
    # TODO: Optimize tracking and filter?
    # tracked_labels.save(video_path.with_name(f"{video_path.stem + '_tracked'}.slp"))

    points_with_score_2d_filtered = get_arr_from_labels(tracked_labels, n_tracks)

    return points_with_score_2d, points_with_score_2d_filtered


def get_arr_from_labels(labels: sleap.Labels, n_tracks: int):
    """
    Get numpy array from sleap labels.

    Args:
        labels: sleap Labels object
        n_tracks: number of instances
    
    Return:
        Array of predicted 2D points with confidence scores. 
            - Shape: (n_tracks, n_frames, n_bodyparts, 3)
            - Final channel: (x, y, score)
    """
    points_with_score_2d = labels.numpy(untracked=False, return_confidence=True) #(n_frames, n_tracks, n_bodyparts, 3)
    points_with_score_2d = np.swapaxes(points_with_score_2d, 0, 1) #(n_tracks, n_frames, n_bodyparts, 3)
    _, n_frames, n_bodyparts, _ = points_with_score_2d.shape

    if len(labels.tracks) < n_tracks:
        padded_points = np.full((n_tracks, n_frames, n_bodyparts, 3), np.nan, dtype="float32")
        for points, track in zip(points_with_score_2d, labels.tracks):
            track_idx = int(track.name)
            padded_points[track_idx] = points
        return padded_points
    
    return points_with_score_2d


class IDTracker(Tracker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.start = False
    
    def track_labels(self, labels):
        tracked_lfs = []
        for lf in labels:
            lf.instances = self.track(lf.instances, t=lf.frame_idx)
            tracked_lfs.append(lf)
        tracked_labels = sleap.Labels(tracked_lfs)
        tracked_labels.videos = sorted(tracked_labels.videos, key=lambda x: x.filename)
        return tracked_labels
    
    def track(self, untracked_instances, t = None):
        previous_t = self.track_matching_queue[-1].t if len(self.track_matching_queue) > 0 else -1
        if t is None:  
            t = previous_t + 1
        else:
            # If the frames are not consecutive, clear the queue
            if previous_t >= 0 and abs(t-self.track_matching_queue[-1].t) > self.track_window:
                self.track_matching_queue.clear()
                self.start = False

        tracked_instances = []
        if untracked_instances:
            if self.pre_cull_function:
                self.pre_cull_function(untracked_instances)

            # If this frame is reliable, we use it as reference
            if len(untracked_instances) == self.target_instance_count \
                and np.mean([inst.score for inst in untracked_instances]) > 0.72:

                self.track_matching_queue.clear()
                self.track_matching_queue.append(MatchedFrameInstances(t, untracked_instances))
                self.start = True
                return untracked_instances
            
            if self.start:
                candidate_instances = self.candidate_maker.get_candidates(self.track_matching_queue, t)

                # Determine matches for untracked instances in current frame.
                frame_matches = FrameMatches.from_candidate_instances(
                    untracked_instances=untracked_instances,
                    candidate_instances=candidate_instances,
                    similarity_function=self.similarity_function,
                    matching_function=self.matching_function,
                    robust_best_instance=self.robust_best_instance,
                )
                self.last_matches = frame_matches

                tracked_instances.extend(self.update_matched_instance_tracks(frame_matches.matches))
                tracked_instances.extend(frame_matches.unmatched_instances)
            else:
                tracked_instances = untracked_instances

        self.track_matching_queue.append(MatchedFrameInstances(t, tracked_instances))

        return tracked_instances


def instance_similarity(ref_instance, query_instance) -> float:
    ref_visible = ~(np.isnan(ref_instance.points_array).any(axis=1))
    dists = np.sum((query_instance.points_array - ref_instance.points_array) ** 2, axis=1)
    similarity = np.nansum(np.exp(-dists/100**2)) / np.sum(ref_visible)

    return similarity