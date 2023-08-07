import os
import cv2
import skvideo.io
import numpy as np
from glob import glob
from tqdm import trange
from marmopose.utils.common import VideoStream
from typing import List, Tuple, Dict

def combine_images(frames2d: List[np.ndarray], frame3d: np.ndarray) -> np.ndarray:
    """
    Combines 2D and 3D images into a single image.
    there are four 2d images, and one 3d image, every 2d image would be resize to 0.5 times of the original size, 
    and then combine as four corners of the original image.
    Then the 3D image would be combined on the right side of the combined 2d image

    Args:
        frames2d: List of 2D images.
        frame3d: 3D image.

    Returns:
        Combined image.
    """
    assert len(frames2d) == 4

    # resize 2d image to 0.5 times of the original size
    height, width, _ = frames2d[0].shape
    new_height, new_width = int(height * 0.5), int(width * 0.5)
    frames2d = [cv2.resize(frame, (new_width, new_height)) for frame in frames2d]
    frame3d = cv2.resize(frame3d, (new_width*3//2, new_height*3//2))

    combined = np.zeros((height, new_width * 3, 3), dtype=np.uint8) + 255

    # TODO: change the shape and position of 3d image
    combined[new_height//4:7*new_height//4, new_width*3//4:new_width*9//4] = frame3d

    combined[:new_height, :new_width] = frames2d[1]
    combined[new_height:, :new_width] = frames2d[0]
    combined[:new_height, 2*new_width:3*new_width] = frames2d[3]
    combined[new_height:, 2*new_width:3*new_width] = frames2d[2]

    return combined


def generate_video_combined(config: Dict, filtered: bool = False, verbose: bool = True) -> None:
    project_dir = config['project_dir']
    video_extension = config['video_extension']

    videos_labeled_2d_dir = config['directory']['videos_labeled_2d_filtered'] if filtered else config['directory']['videos_labeled_2d']
    videos_labeled_3d_dir = config['directory']['videos_labeled_3d_filtered'] if filtered else config['directory']['videos_labeled_3d']

    output_dir = os.path.join(project_dir, videos_labeled_3d_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, 'combined.mp4')

    video2d_paths = sorted(glob(os.path.join(project_dir, videos_labeled_2d_dir, '*.mp4')))
    video3d_path = os.path.join(project_dir, videos_labeled_3d_dir, '3d.mp4')

    caps_2d = []
    for video2d_path in video2d_paths:
        cap = VideoStream(video2d_path, cache_time=5)
        cap.start()
        caps_2d.append(cap)

    cap3d = VideoStream(video3d_path, cache_time=5)
    cap3d.start()
    fps = cap3d.get_params()['fps']
    n_frames = cap3d.get_params()['frames']

    writer = skvideo.io.FFmpegWriter(output_video_path, inputdict={'-framerate': str(fps)}, 
                                     outputdict={'-vcodec': 'libx264', '-pix_fmt': 'yuv420p', '-preset': 'superfast', '-crf': '23'})

    iterator = trange(n_frames, ncols=100, desc=f'Combined Visualizing {os.path.basename(output_video_path)}', unit='frames') if verbose else range(n_frames)

    for frame_idx in iterator:
        frames2d = [cap.read() for cap in caps_2d]
        frames3d = cap3d.read()
        img = combine_images(frames2d, frames3d)
        # img = frames3d
        writer.writeFrame(img)

    for cap in caps_2d:
        cap.stop()
    cap3d.stop()
    writer.close()
