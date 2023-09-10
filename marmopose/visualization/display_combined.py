import cv2
import skvideo.io
import numpy as np
from pathlib import Path
from tqdm import trange
from typing import List, Dict, Any

from marmopose.utils.helpers import VideoStreamThread


def generate_video_combined(config: Dict[str, Any], 
                            points_3d_source: str = 'optimized', 
                            points_2d_source: str = 'reprojected', 
                            verbose: bool = True) -> None:
    """
    Generate a combined video using the specified 3D and 2D points sources.

    Args:
        config: Configuration dictionary containing necessary directory paths and video extensions.
        points_3d_source (optional): Source of 3D points. Should be either 'original' or 'optimized'. Defaults to 'optimized'.
        points_2d_source (optional): Source of 2D points. Should be either 'original', 'filtered', or 'reprojected'. Defaults to 'reprojected'.
        verbose (optional): Whether to show a progress bar during processing. Defaults to True.

    Raises:
        AssertionError: If points_3d_source is not one of the allowed values.
        AssertionError: If points_2d_source is not one of the allowed values.
    """
    assert points_3d_source in ['original', 'optimized'], f'Invalid points_3d_source, must be one of: original, optimized'
    assert points_2d_source in ['original', 'filtered', 'reprojected'], f'Invalid points_2d_source, must be one of: original, filtered, reprojected'

    project_dir = Path(config['directory']['project'])

    video_3d_dir = project_dir / config['directory']['videos_labeled_3d']
    video_3d_path = video_3d_dir / f'{points_3d_source}.mp4'

    video_2d_dir = project_dir / config['directory']['videos_labeled_2d'] / points_2d_source
    video_2d_paths = sorted(video_2d_dir.glob(f"*.{config['video_extension']}"))

    output_video_path = video_3d_dir / f'combined_{points_3d_source}_{points_2d_source}.mp4'

    caps_2d = []
    for video2d_path in video_2d_paths:
        cap = VideoStreamThread(str(video2d_path), cache_time=5, verbose=False)
        cap.start()
        caps_2d.append(cap)

    cap3d = VideoStreamThread(str(video_3d_path), cache_time=5, verbose=False)
    cap3d.start()
    fps, n_frames = cap3d.get_param('fps'), cap3d.get_param('frames')

    writer = skvideo.io.FFmpegWriter(output_video_path, inputdict={'-framerate': str(fps)}, 
                                     outputdict={'-vcodec': 'libx264', '-pix_fmt': 'yuv420p', '-preset': 'superfast', '-crf': '23'})

    iterator = trange(n_frames, ncols=100, desc=f'Combining {output_video_path.stem}', unit='frames') if verbose else range(n_frames)

    for _ in iterator:
        images_2d = np.array([cap.read() for cap in caps_2d])
        image_3d = cap3d.read()
        image_combined = combine_images(images_2d, image_3d)
        writer.writeFrame(image_combined)

    for cap in caps_2d:
        cap.stop()
    cap3d.stop()
    writer.close()


def combine_images(images_2d: np.ndarray, image_3d: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Combine 2D images and a 3D image into a single image.

    Args:
        images_2d : The 2D images to be combined. Shape of (n_images, height, width, 3).
        image_3d : The 3D image to be combined.
        scale (optional): Scale factor for resizing the images. Defaults to 1.0.

    Raises:
        AssertionError: If less than two 2D images are provided.

    Returns:
        The combined image.
    """
    assert len(images_2d) >= 2, 'At least two 2d images are required'

    image_3d = cv2.resize(image_3d, (int(image_3d.shape[1]*scale), int(image_3d.shape[0]*scale)))

    height, width, _ = image_3d.shape
    n_per_column = (len(images_2d)+1)//2
    new_height, new_width = height // n_per_column, width // n_per_column

    images_2d = [cv2.resize(image, (new_width, new_height)) for image in images_2d]

    # TODO: Make it more general
    # This layout is specific designed for videos in this project for best visualization
    image_combined = np.full((height, new_width*3, 3), 255, dtype=np.uint8)

    image_3d = cv2.resize(image_3d, (new_width*3//2, new_height*3//2))
    image_combined[new_height//4:7*new_height//4, new_width*3//4:new_width*9//4] = image_3d

    for i in range(len(images_2d)):
        # r, c = i // 2, i % 2
        r, c = (i+1) % 2, i // 2
        image_combined[r*new_height:(r+1)*new_height, c*(2*new_width):new_width+c*(2*new_width)] = images_2d[i]

    return image_combined
