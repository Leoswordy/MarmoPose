import logging
from pathlib import Path

import cv2
import numpy as np
import skvideo.io
from tqdm import trange

from marmopose.config import Config
from marmopose.utils.data_io import load_points_bboxes_2d_h5
from marmopose.utils.helpers import get_color_list, VideoStreamThread

logger = logging.getLogger(__name__)


class Visualizer2D:
    def __init__(self, config: Config):
        self.init_dir(config)
        self.init_visual_cfg(config)
    
    def init_dir(self, config):
        self.videos_raw_dir = Path(config.sub_directory['videos_raw'])
        self.points_2d_path = Path(config.sub_directory['points_2d']) / 'original.h5'
        self.videos_labeled_2d_dir = Path(config.sub_directory['videos_labeled_2d'])

    def init_visual_cfg(self, config):
        bodyparts = config.animal['bodyparts']
        skeleton = config.visualization['skeleton']
        self.skeleton_indices = [[bodyparts.index(bp) for bp in line] for line in skeleton]
        self.skeleton_color_list = get_color_list(config.visualization['skeleton_cmap'], number=len(skeleton))

        # TODO: The order is specified to be consistent with the marmoset dataset
        colors = get_color_list(config.visualization['track_cmap'])
        new_order = [1, 0, 4, 3, 2]
        self.track_color_list = [colors[i] if i < len(new_order) else colors[i] for i in new_order] + colors[len(new_order):]

    def generate_videos_2d(self):
        # TODO: Visualize specific video, visualize a certain range of frames
        video_paths = sorted(self.videos_raw_dir.glob(f"*.mp4"))
        all_points_with_score_2d, all_bboxes = load_points_bboxes_2d_h5(self.points_2d_path)

        for video_path, points_with_score_2d, bboxes in zip(video_paths, all_points_with_score_2d, all_bboxes):
            output_path = self.videos_labeled_2d_dir / video_path.name
            self.render_video_with_pose(video_path, points_with_score_2d, bboxes, output_path)
    
    def render_video_with_pose(self, video_path: Path, points_with_score_2d: np.ndarray, bboxes: np.ndarray, output_path: Path):
        cap = VideoStreamThread(str(video_path))
        cap.start()
        fps, n_frames = cap.get_param('fps'), cap.get_param('frames')

        writer = skvideo.io.FFmpegWriter(output_path, inputdict={'-framerate': str(fps)}, 
                                         outputdict={'-vcodec': 'libx264', '-pix_fmt': 'yuv420p', '-preset': 'superfast', '-crf': '23'})

        for frame_idx in trange(n_frames, ncols=100, desc=f'2D Visualizing {output_path.stem}', unit='frames'):
            frame = cap.read()
            # TODO: Try to add score to the visualization
            points_2d = points_with_score_2d[:, frame_idx, :, :2] #(n_tracks, n_bodyparts, 2)
            bbox = bboxes[:, frame_idx] #(n_tracks, 4)
            img = self.draw_pose_on_image(frame, points_2d, bbox)
            writer.writeFrame(img)

        cap.stop()
        writer.close()
    
    def draw_pose_on_image(self, img: np.ndarray, points_2d: np.ndarray, bboxes: np.ndarray) -> None:
        for track_idx, (points, bbox) in enumerate(zip(points_2d, bboxes)):
            # Draw bounding boxes
            if not np.any(np.isnan(bbox)):
                x1, y1, x2, y2 = bbox.astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), self.track_color_list[track_idx], 2)
            # Draw points
            valid_points = points[~np.isnan(points).any(axis=-1)].astype(int)
            for x, y in valid_points:
                cv2.circle(img, (x, y), 7, self.track_color_list[track_idx], -1)
            # Draw lines
            self.draw_lines(img, points)

        return img

    def draw_lines(self, img: np.ndarray, points: np.ndarray) -> None:
        for idx, bodypart_indices in enumerate(self.skeleton_indices):
            for a, b in zip(bodypart_indices, bodypart_indices[1:]):
                if np.any(np.isnan(points[[a,b]])):
                    continue
                pa, pb = tuple(map(int, points[a])), tuple(map(int, points[b]))
                cv2.line(img, pa, pb, self.skeleton_color_list[idx], 3)


