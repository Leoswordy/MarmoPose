import os
import cv2
import skvideo.io
import numpy as np
from glob import glob
from tqdm import trange
from matplotlib.pyplot import get_cmap
from typing import List, Tuple, Dict

from marmopose.utils.common import VideoStream
from marmopose.utils.io import load_pose_2d


def draw_line(img: np.ndarray, points: np.ndarray, bodypart_names: List[str], 
              bodyparts: List[str], color: Tuple[int, int, int, int] = (0, 255, 0, 255)) -> None:
    """
    Draws a line on the image connecting given body parts.

    Args:
        img: Image to draw on.
        points: All points of bodyparts.
        bodypart_names: Names of bodyparts to connect.
        all_bodyparts: List of all bodyparts.
        color: Color of the line in RGBA format. Default is green.
    """
    bodypart_indices = [bodyparts.index(bp) for bp in bodypart_names]

    for a, b in zip(bodypart_indices, bodypart_indices[1:]):
        if np.any(np.isnan(points[[a,b]])):
            continue
        pa, pb = tuple(map(int, points[a])), tuple(map(int, points[b]))
        cv2.line(img, pa, pb, color, 3)


def draw_all_lines(img: np.ndarray, points: np.ndarray, scheme: List[List[str]], 
                   bodyparts: List[str], colormap_name: str = 'Set2') -> None:
    """
    Draws all lines defined by the scheme on the image.

    Args:
        img: Image to draw on.
        points: All points of bodyparts.
        scheme: Scheme defines which bodyparts to connect.
        bodyparts: List of all bodyparts.
        colormap_name: Name of the colormap.
    """
    colormap = get_cmap(colormap_name)
    for color_num, bodypart_names in enumerate(scheme):
        color = [int(c) for c in colormap((color_num+1) % len(colormap.colors), bytes=True)]
        draw_line(img, points, bodypart_names, bodyparts, color)


def label_image_with_poses(img: np.ndarray, points: np.ndarray, scheme: List[List[str]], 
                           bodyparts: List[str], colormap_name: str = 'Set3') -> np.ndarray:
    """
    Labels an image with points and lines according to the scheme.

    Args:
        img: Image to label.
        points: All points of bodyparts.
        scheme: Scheme defines which bodyparts to connect.
        bodyparts: List of all bodyparts.
        colormap_name: Name of the colormap.

    Returns:
        np.ndarray: Labelled image.
    """
    colormap = get_cmap(colormap_name)
    for track_idx in range(points.shape[0]):
        draw_all_lines(img, points[track_idx], scheme, bodyparts)

        color = [int(c) for c in colormap(track_idx % len(colormap.colors), bytes=True)[:3]]

        for (x, y) in points[track_idx]:
            if np.isnan(x) or np.isnan(y):
                continue
            x = round(np.clip(x, 1, img.shape[1]-1))
            y = round(np.clip(y, 1, img.shape[0]-1))
            cv2.circle(img, (x, y), 7, color, -1)

    return img


def write_poses_on_video(config: Dict, poses_filepath: str, input_video_path: str, output_video_path: str, verbose: bool) -> None:
    """
    Overlay poses on video frames and write it into a new video.

    Args:
        config: The configuration dictionary.
        poses_filepath: File path of the pose data.
        input_video_path: File path of the input video.
        output_video_path: File path of the output video.
        verbose: Verbose mode.
    """
    bodyparts = config['bodyparts']
    scheme = config['visualization']['scheme']

    all_points_scores, _ = load_pose_2d(poses_filepath)
    all_points = all_points_scores[..., :2] #(n_tracks, n_frames, n_bodyparts, 2)

    cap = VideoStream(input_video_path)
    cap.start()
    fps = cap.get_params()['fps']
    n_frames = cap.get_params()['frames']
    writer = skvideo.io.FFmpegWriter(output_video_path, inputdict={'-framerate': str(fps)}, 
                                     outputdict={'-vcodec': 'libx264', '-pix_fmt': 'yuv420p', '-preset': 'superfast', '-crf': '23'})

    iterator = trange(n_frames, ncols=100, desc=f'2D Visualizing {os.path.basename(output_video_path)}', unit='frames') if verbose else range(n_frames)

    for frame_idx in iterator:
        frame = cap.read()
        points = all_points[:, frame_idx]

        img = label_image_with_poses(frame, points, scheme, bodyparts)
        writer.writeFrame(img)

    cap.stop()
    writer.close()


def generate_video_2d(config: Dict, filtered: bool = False, verbose: bool = True) -> None:
    """
    Generate a video with 2D poses based on a given configuration.

    Args:
        config: The configuration dictionary.
        filtered: If True, use filtered poses.
        verbose: Verbose mode. Default is True.
    """
    project_dir = config['project_dir']
    video_extension = config['video_extension']
    videos_raw_dir = config['directory']['videos_raw']

    poses_2d_dir = config['directory']['poses_2d_filtered'] if filtered else config['directory']['poses_2d']
    videos_labeled_2d_dir = config['directory']['videos_labeled_2d_filtered'] if filtered else config['directory']['videos_labeled_2d']

    output_dir = os.path.join(project_dir, videos_labeled_2d_dir)
    os.makedirs(output_dir, exist_ok=True)

    poses_filepaths = sorted(glob(os.path.join(project_dir, poses_2d_dir, '*.h5')))

    for poses_filepath in poses_filepaths:
        basename = os.path.splitext(os.path.basename(poses_filepath))[0]
        input_video_path = os.path.join(project_dir, videos_raw_dir, basename+'.'+video_extension)
        output_video_path = os.path.join(output_dir, basename+'.'+video_extension)
        
        write_poses_on_video(config, poses_filepath, input_video_path, output_video_path, verbose)