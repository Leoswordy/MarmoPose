import numpy as np
from pathlib import Path
from typing import Dict, Any

import sleap

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
    videos_raw_dir = project_dir / config['directory']['videos_raw']
    points_2d_path = project_dir / config['directory']['points_2d'] / 'original.h5'
    points_2d_path.parent.mkdir(parents=True, exist_ok=True)
    points_2d_filtered_path = project_dir / config['directory']['points_2d'] / 'filtered.h5'
    points_2d_filtered_path.parent.mkdir(parents=True, exist_ok=True)
    # Find all videos in the raw videos directory
    video_paths = sorted(videos_raw_dir.glob(f"*.{config['video_extension']}"))
    
    # Predict 2D points for each video and save to an HDF5 file
    for video_path in video_paths:
        points_with_score_2d = predict_points_2d(video_path, model_dir, batch_size, verbose)
        save_points_2d_h5(points=points_with_score_2d,
                          name=video_path.stem, 
                          file_path=points_2d_path, 
                          verbose=verbose)
        
        points_with_score_2d_filtered = filter_data(points_with_score_2d, threshold=config['filter']['threshold'])
        save_points_2d_h5(points=points_with_score_2d_filtered,
                          name=video_path.stem, 
                          file_path=points_2d_filtered_path, 
                          verbose=verbose)


def predict_points_2d(video_path: Path, model_dir: str, batch_size: int = 4 , verbose: bool = True) -> np.ndarray:
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
    
    points_with_score_2d = labels.numpy(untracked=False, return_confidence=True) #(n_frames, n_tracks, n_bodyparts, 3)
    points_with_score_2d = np.swapaxes(points_with_score_2d, 0, 1) #(n_tracks, n_frames, n_bodyparts, 3)

    return points_with_score_2d
