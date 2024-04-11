import logging
from pathlib import Path
from typing import Any, List, Tuple

import cv2
import numpy as np
from mayavi import mlab
import skvideo.io
from tqdm import trange

from marmopose.config import Config
from marmopose.utils.data_io import load_points_3d_h5
from marmopose.utils.helpers import get_color_list, VideoStreamThread

mlab.options.offscreen = True
logger = logging.getLogger(__name__)


class Visualizer3D:
    def __init__(self, config: Config):
        self.init_dir(config)
        self.init_visual_cfg(config)
    
    def init_dir(self, config):
        self.points_3d_path = Path(config.sub_directory['points_3d']) / 'original.h5'
        self.video_labeled_3d_path = Path(config.sub_directory['videos_labeled_3d']) / 'original.mp4'
        self.videos_2d_dir = Path(config.sub_directory['videos_labeled_2d'])
        self.video_combined_path = Path(config.sub_directory['videos_labeled_3d']) / 'combined.mp4'
    
    def init_visual_cfg(self, config):
        bodyparts = config.animal['bodyparts']
        skeleton = config.visualization['skeleton']
        self.skeleton_indices = [[bodyparts.index(bp) for bp in line] for line in skeleton]
        self.skeleton_color_list = get_color_list(config.visualization['skeleton_cmap'], number=len(skeleton), cvtInt=False)

        # TODO: The order is specified to be consistent with the marmoset dataset
        colors = get_color_list(config.visualization['track_cmap'], cvtInt=False)
        new_order = [1, 0, 4, 3, 2]
        self.track_color_list = [colors[i] if i < len(new_order) else colors[i] for i in new_order] + colors[len(new_order):]
    
    # def generate_video_3d(self, source='original', fps: int=25):
    #     assert source in ['original', 'optimized'], f'Invalid data source: {source}'
    #     if source == 'optimized':
    #         self.points_3d_path = self.points_3d_path.with_name('optimized.h5')
    #         self.video_labeled_3d_path = self.video_labeled_3d_path.with_name('optimized.mp4')

    #     all_points_3d = load_points_3d_h5(self.points_3d_path)
    #     n_tracks, n_frames, n_bodyparts, _ = all_points_3d.shape

    #     fig, mlab_points, mlab_lines = self.initialize_3d(n_tracks, n_bodyparts)

    #     writer = skvideo.io.FFmpegWriter(self.video_labeled_3d_path, inputdict={'-framerate': str(fps)},
    #                                 outputdict={'-vcodec': 'libx264', '-pix_fmt': 'yuv420p', '-preset': 'superfast', '-crf': '23'})

    #     for frame_idx in trange(n_frames, ncols=100, desc='3D Visualizing... ', unit='frames'):
    #         img = self.get_image_3d(fig, all_points_3d[:, frame_idx], mlab_points, mlab_lines, show_lines=True)
    #         writer.writeFrame(img)

    #     writer.close()
    #     mlab.close(fig)

    def generate_video_3d(self, source: str = 'original', fps: int = 25, start_frame_idx: int = 0, end_frame_idx: int = None):
        assert source in ['original', 'optimized'], f'Invalid data source: {source}'
        if source == 'optimized':
            self.points_3d_path = self.points_3d_path.with_name('optimized.h5')
            self.video_labeled_3d_path = self.video_labeled_3d_path.with_name('optimized.mp4')

        all_points_3d = load_points_3d_h5(self.points_3d_path)
        n_tracks, n_frames, n_bodyparts, _ = all_points_3d.shape

        # Adjust end_frame_idx if it is None or beyond the total number of frames
        end_frame_idx = min(end_frame_idx if end_frame_idx is not None else n_frames, n_frames)
        logger.info(f'Generating 3D video from frame {start_frame_idx} to {end_frame_idx}')

        fig, mlab_points, mlab_lines = self.initialize_3d(n_tracks, n_bodyparts)

        writer = skvideo.io.FFmpegWriter(self.video_labeled_3d_path, inputdict={'-framerate': str(fps)},
                                         outputdict={'-vcodec': 'libx264', '-pix_fmt': 'yuv420p', '-preset': 'superfast', '-crf': '23'})

        for frame_idx in trange(start_frame_idx, end_frame_idx, ncols=100, desc='3D Visualizing... ', unit='frames'):
            img = self.get_image_3d(fig, all_points_3d[:, frame_idx], mlab_points, mlab_lines, show_lines=True)
            writer.writeFrame(img)

        writer.close()
        mlab.close(fig)
    
    def get_image_3d(self, fig: mlab.figure, 
                 all_points: np.ndarray, 
                 mlab_points: List[mlab.points3d], 
                 mlab_lines: List[mlab.plot3d],
                 show_lines: bool = True) -> np.ndarray:
        """
        Capture a 3D image frame with updated pose points and lines.
        
        Args:
            fig: Mayavi figure object for rendering.
            all_points: Array of shape (n_tracks, n_bodyparts, 3) containing points in 3D.
            mlab_points: List of Mayavi point objects for each track.
            mlab_lines: List of Mayavi line objects for each track.
            skeleton_indices: Indices of body parts that are connected by lines in the skeleton.
            show_lines (optional): Whether to show skeleton lines. Defaults to True.
        
        Returns:
            Captured image frame.
        """
        fig.scene.disable_render = True
        for points, mlab_point, mlab_line in zip(all_points, mlab_points, mlab_lines):
            mlab_point.mlab_source.points = points
            if show_lines: self.update_lines(mlab_line, points)
        fig.scene.disable_render = False
        # mlab.show()
        return mlab.screenshot(antialiased=True)
    
    def update_lines(self, lines: List[Any], points: np.ndarray) -> None:
        for line, bodyparts_indices in zip(lines, self.skeleton_indices):
            line.mlab_source.points = points[bodyparts_indices]

    def initialize_3d(self, n_tracks: int, n_bodyparts: int,
                  axes: np.ndarray = None, scale: float = 1.0, 
                  room_dimensions: List[int] = [730, 1030, 860, 30]) -> Tuple[mlab.figure, List[mlab.points3d], List[mlab.plot3d]]:
        fig = mlab.figure(bgcolor=(1, 1, 1), size=(int(1920*scale), int(1080*scale)))
        fig.scene.anti_aliasing_frames = 8
        mlab.clf()

        # TODO: How to define room dimensions?****************************************************************
        # Draw room with grids
        # mlab.view(135, 120)
        self.draw_room_grids(*room_dimensions)

        # Draw axes
        scale_factor = np.max(room_dimensions)//50
        if axes is not None:
            mlab.points3d(axes[:, 0], axes[:, 1], axes[:, 2], color=(0.8, 0.8, 0.8), scale_factor=scale_factor)
        #****************************************************************************************************

        # Initialize points and lines
        mlab_points, mlab_lines = [], []
        for track_idx in range(n_tracks):
            points = np.zeros((n_bodyparts, 3))

            mlab_points.append(mlab.points3d(points[:, 0], points[:, 1], points[:, 2], color=self.track_color_list[track_idx], scale_factor=scale_factor))
            mlab_lines.append(self.draw_lines(points))

        return fig, mlab_points, mlab_lines
    
    @staticmethod
    def draw_room_grids(width: int, length: int, height: int, grid_size: int) -> None:
        """
        Draw grids on the room floor and walls.
        
        Args:
            width: Width of the room.
            length: Length of the room.
            height: Height of the room.
            grid_size: Size of the grid squares.
        """
        col = (1, 1, 1)
        x, y = np.mgrid[:width:grid_size, :length:grid_size]
        mlab.surf(x, y, np.zeros(x.shape), representation='wireframe', color=col, line_width=0.5)
        z, y = np.mgrid[:height:grid_size, :length:grid_size]
        mlab.surf(np.zeros(y.shape), y, z, representation='wireframe', color=col, line_width=0.5)
        x, z = np.mgrid[:width:grid_size, :height:grid_size]
        mlab.surf(x, np.zeros(z.shape), z, representation='wireframe', color=col, line_width=0.5)
    
    def draw_lines(self, points: np.ndarray) -> List[mlab.plot3d]:
        """
        Draw all lines connecting body parts based on the skeleton configuration.
        
        Args:
            points: Array of 3D coordinates of body parts.
            skeleton_indices: List of index pairs indicating which body parts to connect.
            skeleton_color_list: List of RGB colors for each skeleton line.
        
        Returns:
            List of created line objects.
        """
        lines = []
        for idx, bodyparts_indices in enumerate(self.skeleton_indices):
            line = mlab.plot3d(points[bodyparts_indices, 0], points[bodyparts_indices, 1], points[bodyparts_indices, 2],
                            np.ones(len(bodyparts_indices)), reset_zoom=False,
                            color=self.skeleton_color_list[idx], tube_radius=None, line_width=5)
            lines.append(line)

        return lines
    
    def generate_video_combined(self, source='original'):
        assert source in ['original', 'optimized'], f'Invalid data source: {source}'
        if source == 'optimized':
            self.video_labeled_3d_path = self.video_labeled_3d_path.with_name('optimized.mp4')
            self.video_combined_path = self.video_combined_path.with_name('combined_optimized.mp4')

        caps_2d = []
        for video2d_path in sorted(self.videos_2d_dir.glob(f"*.mp4")):
            cap = VideoStreamThread(str(video2d_path), cache_time=5)
            cap.start()
            caps_2d.append(cap)

        cap3d = VideoStreamThread(str(self.video_labeled_3d_path), cache_time=5)
        cap3d.start()
        fps, n_frames = cap3d.get_param('fps'), cap3d.get_param('frames')

        writer = skvideo.io.FFmpegWriter(self.video_combined_path, inputdict={'-framerate': str(fps)}, 
                                        outputdict={'-vcodec': 'libx264', '-pix_fmt': 'yuv420p', '-preset': 'superfast', '-crf': '23'})

        for _ in trange(n_frames, ncols=100, desc=f'Combining {self.video_combined_path.stem}', unit='frames'):
            images_2d = np.array([cap.read() for cap in caps_2d])
            image_3d = cap3d.read()
            image_combined = self.combine_images(images_2d, image_3d)
            writer.writeFrame(image_combined)

        for cap in caps_2d:
            cap.stop()
        cap3d.stop()
        writer.close()

    @staticmethod
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
        # TODO: Rearrange the layout of the combined image
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