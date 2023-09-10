import cv2
import skvideo.io
import numpy as np
from tqdm import trange
from pathlib import Path
from typing import Dict, Any, List, Tuple

from marmopose.utils.data_io import load_points_2d_h5
from marmopose.utils.helpers import get_color_list, VideoStreamThread


def generate_video_2d(config: Dict[str, Any], points_2d_source: str = 'original', verbose: bool = True) -> None:
    """Generate 2D videos with labeled body parts and skeleton.
    
    Args:
        config: Configuration dictionary containing project parameters.
        points_2d_source (optional): Source of 2D points. Can be 'original', 'filtered', or 'reprojected'. Defaults to 'original'.
        verbose (optional): If True, prints detailed logs. Defaults to True.
        
    Raises:
        AssertionError: If points_2d_source is invalid.
    """
    assert points_2d_source in ['original', 'filtered', 'reprojected'], f'Invalid points_2d_source, must be one of: original, filtered, reprojected'

    project_dir = Path(config['directory']['project'])
    videos_raw_dir = project_dir / config['directory']['videos_raw']
    videos_labeled_2d_dir = project_dir / config['directory']['videos_labeled_2d'] / points_2d_source
    points_2d_path = project_dir / config['directory']['points_2d'] / f'{points_2d_source}.h5'
    videos_labeled_2d_dir.mkdir(parents=True, exist_ok=True)

    bodyparts = config['animal']['bodyparts']
    skeleton = config['visualization']['skeleton']
    skeleton_indices = [[bodyparts.index(bp) for bp in line] for line in skeleton]

    track_color_list = get_color_list(config['visualization']['track_cmap'], config['animal']['number'])
    skeleton_color_list = get_color_list(config['visualization']['skeleton_cmap'], len(skeleton_indices))

    video_paths = sorted(videos_raw_dir.glob(f"*.{config['video_extension']}"))
    all_points_with_score_2d = load_points_2d_h5(points_2d_path, verbose=verbose)

    for video_path, points_with_score_2d in zip(video_paths, all_points_with_score_2d):
        output_path = videos_labeled_2d_dir / video_path.name
        render_video_with_pose(video_path, points_with_score_2d, skeleton_indices, output_path,
                               track_color_list, skeleton_color_list, verbose=verbose)


def render_video_with_pose(video_path: Path, points_with_score_2d: np.ndarray, skeleton_indices: List[List[int]], output_path: Path, 
                           track_color_list: List[Tuple[int]], skeleton_color_list: List[Tuple[int]], verbose: bool = True) -> None:
    """Render video with labeled body parts and skeleton.
    
    Args:
        video_path: The path to the raw video file.
        points_with_score_2d: The 2D coordinates of keypoints. Shape of (n_tracks, n_frames, n_bodyparts, 3), final channel (x, y, score).
        skeleton_indices: Indices for drawing skeleton lines.
        output_path: The path for saving the output video.
        track_color_list: A list of RGB color tuples for each track.
        skeleton_color_list: A list of RGB color tuples for each skeleton line.
        verbose (optional): If True, prints detailed logs. Defaults to True.
    """
    cap = VideoStreamThread(str(video_path))
    cap.start()
    fps, n_frames = cap.get_param('fps'), cap.get_param('frames')

    writer = skvideo.io.FFmpegWriter(output_path, inputdict={'-framerate': str(fps)}, 
                                    outputdict={'-vcodec': 'libx264', '-pix_fmt': 'yuv420p', '-preset': 'superfast', '-crf': '23'})
    
    iterator = trange(n_frames, ncols=100, desc=f'2D Visualizing {output_path.stem}', unit='frames') if verbose else range(n_frames)

    for frame_idx in iterator:
        frame = cap.read()
        # TODO: Try to add score to the visualization
        points_2d = points_with_score_2d[:, frame_idx, :, :2] #(n_tracks, n_bodyparts, 2)
        img = label_image_with_pose(frame, points_2d, skeleton_indices, track_color_list, skeleton_color_list)
        writer.writeFrame(img)

    cap.stop()
    writer.close()


def label_image_with_pose(img: np.ndarray, points_2d: np.ndarray, skeleton_indices: List[List[int]], 
                          track_color_list: List[Tuple[int]], skeleton_color_list: List[Tuple[int]]) -> np.ndarray:
    """Labels an image frame with body parts and skeleton.
    
    Args:
        img: The input image frame.
        points_2d: The 2D coordinates of keypoints. Shape of (n_tracks, n_bodyparts, 2).
        skeleton_indices: Indices for drawing skeleton lines.
        track_color_list: A list of RGB color tuples for each track.
        skeleton_color_list: A list of RGB color tuples for each skeleton line.
        
    Returns:
        The labeled image.
    """
    for track_idx, points in enumerate(points_2d):
        # Draw points
        valid_points = points[~np.isnan(points).any(axis=-1)].astype(int)
        for x, y in valid_points:
            cv2.circle(img, (x, y), 7, track_color_list[track_idx], -1)
        # Draw lines
        draw_lines(img, points, skeleton_indices, skeleton_color_list)

    return img


def draw_lines(img: np.ndarray, points: np.ndarray, skeleton_indices: List[List[int]], 
               skeleton_color_list: List[Tuple[int]]) -> None:
    """Draws all skeletal lines between body parts on an image frame.
    
    Args:
        img: The input image frame.
        points: The 2D coordinates of keypoints. Shape of (n_bodyparts, 2).
        skeleton_indices: Indices for drawing skeleton lines.
        skeleton_color_list: A list of RGB color tuples for each skeleton line.
    """
    for idx, bodypart_indices in enumerate(skeleton_indices):
        for a, b in zip(bodypart_indices, bodypart_indices[1:]):
            if np.any(np.isnan(points[[a,b]])):
                continue
            pa, pb = tuple(map(int, points[a])), tuple(map(int, points[b]))
            cv2.line(img, pa, pb, skeleton_color_list[idx], 3)