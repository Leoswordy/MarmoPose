import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np
import open3d as o3d
import skvideo.io
from tqdm import trange

from marmopose.config import Config
from marmopose.utils.data_io import load_points_3d_h5
from marmopose.utils.helpers import get_color_list, MultiVideoCapture

logger = logging.getLogger(__name__)


WIDTH_3D = 960
HEIGHT_3D = 1080


class Visualizer3D:
    def __init__(self, config: Config):
        self.init_dir(config)
        self.init_visual_cfg(config)
    
    def init_dir(self, config):
        self.config = config
        self.points_3d_path = Path(config.sub_directory['points_3d']) / 'original.h5'
        self.video_labeled_3d_path = Path(config.sub_directory['videos_labeled_3d']) / 'original.mp4'
        self.videos_2d_dir = Path(config.sub_directory['videos_labeled_2d'])
    
    def init_visual_cfg(self, config):
        bodyparts = config.animal['bodyparts']
        skeleton = config.visualization['skeleton']
        self.skeleton_indices = [[bodyparts.index(bp) for bp in line] for line in skeleton]
        self.skeleton_color_list = get_color_list(config.visualization['skeleton_cmap'], number=len(skeleton), cvtInt=False)

        # TODO: The order is specified to be consistent with the marmoset dataset
        colors = get_color_list(config.visualization['track_cmap'], cvtInt=False)
        new_order = [1, 0, 4, 3, 2]
        self.track_color_list = [colors[i] if i < len(new_order) else colors[i] for i in new_order] + colors[len(new_order):]
    
    def generate_video_3d(self, source_3d: str = 'original', start_frame_idx: int = 0, end_frame_idx: int = None, video_type: str = 'composite', fps: int = 25):
        assert source_3d in ['original', 'optimized'], f'Invalid data source: {source_3d}'
        assert video_type in ['3d', 'composite'], f'Invalid video type: {video_type}'

        if source_3d == 'optimized':
            self.points_3d_path = self.points_3d_path.with_name('optimized.h5')
        self.video_labeled_3d_path = self.video_labeled_3d_path.with_name(f'{source_3d}_{video_type}.mp4')
        if not (start_frame_idx == 0 and end_frame_idx is None):
            self.video_labeled_3d_path = self.video_labeled_3d_path.with_name(f'{self.video_labeled_3d_path.stem}_{start_frame_idx}_{end_frame_idx}.mp4')
        desc_info = f'Visualizing {video_type}... '

        all_points_3d = load_points_3d_h5(self.points_3d_path)
        n_tracks, n_frames, n_bodyparts, _ = all_points_3d.shape

        self.initialize_3d(n_tracks, n_bodyparts)

        writer = skvideo.io.FFmpegWriter(self.video_labeled_3d_path, inputdict={'-framerate': str(fps)},
                                        outputdict={'-vcodec': 'libx264', '-pix_fmt': 'yuv420p', '-preset': 'superfast', '-crf': '23'})

        end_frame_idx = min(end_frame_idx if end_frame_idx is not None else n_frames, n_frames)
        logger.info(f'Generating {video_type} video from frame {start_frame_idx} to {end_frame_idx}')

        if video_type == 'composite':
            video_paths = sorted([str(p) for p in self.videos_2d_dir.glob("*.mp4")])
            mvc = MultiVideoCapture(video_paths, video_paths, do_cache=True, simulate_live=False, start_frame_idx=start_frame_idx, end_frame_idx=end_frame_idx)
            mvc.start()

        for frame_idx in trange(start_frame_idx, end_frame_idx, ncols=100, desc=desc_info, unit='frames'):
            img_3d = self.get_image_3d(all_points_3d[:, frame_idx]) # ~19ms
            if video_type == '3d':
                img_out = img_3d
            elif video_type == 'composite': 
                images_2d = mvc.get_next_frames()
                images_2d = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images_2d]
                img_out = self.combine_images(images_2d, img_3d) # ~10ms
            
            writer.writeFrame(img_out)

        writer.close()
        self.vis.destroy_window()

        if video_type == 'composite':
            mvc.stop()

    def initialize_3d(self, n_tracks: int, n_bodyparts: int, width: int = WIDTH_3D, height: int = HEIGHT_3D, room_dimensions: List[int] = [720, 1050, 840, 30]):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=width, height=height, visible=False)

        opt = self.vis.get_render_option()
        opt.point_size = 7
        opt.light_on = False

        self.draw_room_grids(self.vis, *room_dimensions)

        lines_list = []
        for indices in self.skeleton_indices:
            lines_list.extend([[indices[i], indices[i+1]] for i in range(len(indices)-1)])

        line_colors = []
        for idx, indices in enumerate(self.skeleton_indices):
            color = self.skeleton_color_list[idx]
            color = np.array(color) / 255.0 if np.max(color) > 1 else np.array(color)

            line_colors.extend([color for _ in range(len(indices)-1)])

        self.point_clouds, self.line_sets = [], []
        for track_idx in range(n_tracks):
            points = o3d.geometry.PointCloud()
            points.points = o3d.utility.Vector3dVector(np.zeros((n_bodyparts, 3)))
            color = self.track_color_list[track_idx]
            color = np.array(color) / 255.0 if np.max(color) > 1 else np.array(color)
            points.colors = o3d.utility.Vector3dVector(np.tile(color, (n_bodyparts, 1)))
            self.vis.add_geometry(points)
            self.point_clouds.append(points)
            
            lines = o3d.geometry.LineSet()
            lines.points = points.points
            lines.lines = o3d.utility.Vector2iVector(lines_list)
            lines.colors = o3d.utility.Vector3dVector(line_colors)
            self.vis.add_geometry(lines)
            self.line_sets.append(lines)

        ctr = self.vis.get_view_control()
        ctr.set_lookat([0, 200, 0])
        ctr.set_front([1, 1, 1.1])
        ctr.set_up([0, 0, 1])
        ctr.set_zoom(1.25)
    
    def get_image_3d(self, all_points: np.ndarray) -> np.ndarray:
        for points, point_cloud, line_set in zip(all_points, self.point_clouds, self.line_sets):
            point_cloud.points = o3d.utility.Vector3dVector(points)
            self.vis.update_geometry(point_cloud)

            line_set.points = point_cloud.points
            self.vis.update_geometry(line_set)

        self.vis.poll_events()
        self.vis.update_renderer()

        img = np.asarray(self.vis.capture_screen_float_buffer(False))
        img = (img * 255).astype(np.uint8)
        return img

    @staticmethod
    def draw_room_grids(vis: o3d.visualization.Visualizer, width: int, length: int, height: int, grid_size: int) -> None:
        """
        Draw grids on the room floor and walls.

        Args:
            vis: Open3D visualizer.
            width: Width of the room along the x-axis.
            length: Length of the room along the y-axis.
            height: Height of the room along the z-axis.
            grid_size: Size of the grid squares.
        """
        points = []
        lines = []
        idx = 0

        grid_color = [0.85, 0.85, 0.85]
        colors = []

        ##### Floor Grid (z = 0 plane) #####
        # Horizontal lines (along y-axis)
        for x in range(0, width+grid_size, grid_size):
            points.append([x, 0, 0])
            points.append([x, length, 0])
            lines.append([idx, idx + 1])
            colors.append(grid_color)
            idx += 2

        # Vertical lines (along x-axis)
        for y in range(0, length+grid_size, grid_size):
            points.append([0, y, 0])
            points.append([width, y, 0])
            lines.append([idx, idx + 1])
            colors.append(grid_color)
            idx += 2

        ##### Left Wall Grid (x = 0 plane) #####
        # Horizontal lines (along z-axis)
        for y in range(0, length+grid_size, grid_size):
            points.append([0, y, 0])
            points.append([0, y, height])
            lines.append([idx, idx + 1])
            colors.append(grid_color)
            idx += 2

        # Vertical lines (along y-axis)
        for z in range(0, height+grid_size, grid_size):
            points.append([0, 0, z])
            points.append([0, length, z])
            lines.append([idx, idx + 1])
            colors.append(grid_color)
            idx += 2

        ##### Back Wall Grid (y = 0 plane) #####
        # Horizontal lines (along z-axis)
        for x in range(0, width+grid_size, grid_size):
            points.append([x, 0, 0])
            points.append([x, 0, height])
            lines.append([idx, idx + 1])
            colors.append(grid_color)
            idx += 2

        # Vertical lines (along x-axis)
        for z in range(0, height+grid_size, grid_size):
            points.append([0, 0, z])
            points.append([width, 0, z])
            lines.append([idx, idx + 1])
            colors.append(grid_color)
            idx += 2

        # Create LineSet for room grids
        room_grid = o3d.geometry.LineSet()
        room_grid.points = o3d.utility.Vector3dVector(points)
        room_grid.lines = o3d.utility.Vector2iVector(lines)
        room_grid.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(room_grid)

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
        assert len(images_2d) >= 2, 'At least two 2d images are required'

        image_3d = cv2.resize(image_3d, (int(image_3d.shape[1]*scale), int(image_3d.shape[0]*scale)))

        height_3d, width_3d, _ = image_3d.shape
        n_per_column = (len(images_2d)+1)//2

        height_2d, width_2d, _ = images_2d[0].shape
        new_height_2d = height_3d // n_per_column
        new_width_2d = int(new_height_2d * width_2d / height_2d)

        images_2d = [cv2.resize(image, (new_width_2d, new_height_2d)) for image in images_2d]

        image_combined = np.full((height_3d, width_3d+2*new_width_2d, 3), 255, dtype=np.uint8)

        image_combined[:, new_width_2d:new_width_2d+width_3d] = image_3d
        for i in range(len(images_2d)):
            r, c = (i+1) % 2, i // 2
            image_combined[r*new_height_2d:(r+1)*new_height_2d, c*(new_width_2d+width_3d):new_width_2d+c*(new_width_2d+width_3d)] = images_2d[i]

        return image_combined

    def capture_high_res_image(self, source_3d: str = 'original', frame_idx: int = 0, output_path: str = None, scale=8) -> None:
        if source_3d == 'optimized':
            self.points_3d_path = self.points_3d_path.with_name('optimized.h5')
    
        all_points_3d = load_points_3d_h5(self.points_3d_path)
        n_tracks, n_frames, n_bodyparts, _ = all_points_3d.shape

        self.initialize_3d(n_tracks, n_bodyparts, width=WIDTH_3D*scale, height=HEIGHT_3D*scale)

        img = self.get_image_3d(all_points_3d[:, frame_idx])

        from PIL import Image
        image = Image.fromarray(img)
        if output_path is None:
            output_path = Path(self.config.project_path) / 'images' / f'frame_{frame_idx}.png'
            output_path.parent.mkdir(parents=True, exist_ok=True)
                    
        image.save(output_path, format="png", dpi=(300, 300))

        self.vis.destroy_window()
