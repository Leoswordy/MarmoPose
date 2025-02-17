import cv2
import time
import logging
import numpy as np
import skvideo.io
from pathlib import Path
from typing import List, Tuple
from multiprocessing import Process, Queue

from marmopose.config import Config
from marmopose.processing.prediction import Predictor
from marmopose.processing.triangulation import Reconstructor3D, filter_outliers_by_skeleton
from marmopose.utils.helpers import Timer, MultiVideoCapture
from marmopose.visualization.display_3d import Visualizer3D
from marmopose.visualization.display_2d import Visualizer2D

logger = logging.getLogger(__name__)


class PredictProcess(Process):
    def __init__(self, 
                 config: Config,
                 camera_paths: List[str], 
                 camera_names: List[str],
                 data_queue: Queue,
                 stop_event,
                 display_2d: bool,
                 simulate_live: bool = False):
        super().__init__()
        self.camera_paths = camera_paths
        self.camera_names = camera_names
        self.config = config

        self.data_queue = data_queue
        self.stop_event = stop_event

        self.display_2d = display_2d

        self.simulate_live = simulate_live
    
    def run(self):
        # Must init on the new process, not in the init method
        self.predictor = Predictor(self.config, batch_size=1)
        self.reconstructor_3d = Reconstructor3D(self.config)
        self.visualizer_2d = Visualizer2D(self.config)
    
        self.mvc = MultiVideoCapture(paths = self.camera_paths, 
                                     names = self.camera_names, 
                                     simulate_live = self.simulate_live, 
                                     output_dir = self.config.sub_directory['videos_raw'])
        self.mvc.start()

        timer = Timer(name='Predict', output_path=f'{self.config.project_path}/log/predict_latency.npz').start()
        while not self.stop_event.is_set():
            images = self.mvc.get_next_frames()

            if images is None:
                time.sleep(0.01)
                continue
            timer.record('Read')

            if self.should_skip_processing():
                # NOTE: Skip processing if inference speed is slower than cache speed
                n_cams = len(self.camera_paths)
                n_tracks = self.config.animal['n_tracks']
                n_bodyparts = len(self.config.animal['bodyparts'])
                points_with_score_2d_batch = np.full((n_cams, n_tracks, n_bodyparts, 3), np.nan)
                bboxes_batch = np.full((n_cams, n_tracks, 4), np.nan)
                all_points_3d = np.full((n_tracks, 1, n_bodyparts, 3), np.nan)
            else:
                points_with_score_2d_batch, bboxes_batch = self.predictor.predict_image_batch(images)
                timer.record('Predict')

                all_points_3d = self.triangulate(points_with_score_2d_batch)
                timer.record('Triangulate')

            processed_images = self.annotate_and_downsample_images(images, points_with_score_2d_batch, bboxes_batch) if self.display_2d else None
            timer.record('Process') 
            
            data = {
                'points_2d': points_with_score_2d_batch, # (n_camsx1, n_tracks, n_bodyparts, 3)
                'bboxes': bboxes_batch, # (n_camsx1, n_tracks, 4)
                'points_3d': all_points_3d, # (n_tracks, n_frames=1, n_bodyparts, 3)
                'images_2d': processed_images # (n_camsx1, h, w, 3)
            }

            self.data_queue.put(data)
            timer.record('Send')

            timer.show()
        timer.show_avg(exclude=['Read'])

        self.mvc.stop()
        logger.info("Predict Process Finished")

    def annotate_and_downsample_images(self, images: List[np.ndarray], points_with_score_2d_batch: np.ndarray, bboxes_batch: np.ndarray, downsample_scale: float = 0.5) -> List[np.ndarray]:
        """ Annotate 2D poses and bounding boxes on images and downsample them.

        Args:
            images: A list of images.
            points_with_score_2d_batch: Inferred 2D points with shape (n_camsx1, n_tracks, n_bodyparts, 3).
            bboxes_batch: Bounding boxes with shape (n_camsx1, n_tracks, 4).
            downsample_scale: The scale to downsample images.
        
        Returns:
            A list of annotated and downsampled images.
        """
        original_width, original_height = images[0].shape[1], images[0].shape[0]
        new_shape = (int(original_width*downsample_scale), int(original_height*downsample_scale))

        processed_images = []
        for points_with_score_2d, bbox, image in zip(points_with_score_2d_batch, bboxes_batch, images):
            # TODO: Draw 2D poses and bounding boxes on images takes ~4ms, ignore it if not necessary
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            points_2d = points_with_score_2d[:, :, :2]
            image = self.visualizer_2d.draw_pose_on_image(image, points_2d, bbox)

            # Must resize for faster inter-multiprocess communication!
            # Otherwise serialize and deserialize data between processes could be shockingly slow!
            # If you don't need to display the 2D image, it won't matter.
            # TODO: We should find a better way to organize the code.
            image = cv2.resize(image, new_shape)

            processed_images.append(image)

        return processed_images

    def triangulate(self, all_points_with_score_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computes 3D points and scores using triangulation.

        Args:
            all_points_with_scores_2d: Inferred points with shape (n_cams, n_tracks, n_bodyparts, 3).

        Returns:
            A tuple containing 3D points and scores.
                - all_points_3d: 3D points with shape (n_tracks, n_frames=1, n_bodyparts, 3).
        """
        all_points_with_score_2d = np.array(all_points_with_score_2d)
        all_points_3d = self.reconstructor_3d.triangulate_frame(all_points_with_score_2d, ransac=False)
        all_points_3d = np.expand_dims(all_points_3d, axis=1)
        all_points_3d = filter_outliers_by_skeleton(all_points_3d, 150)
        
        return all_points_3d
    
    def should_skip_processing(self) -> bool:
        """Decides whether to skip processing based on queue sizes."""
        queue_sizes = self.mvc.get_qsizes()
        logger.debug(f"[Predict] - Cached Frame: {queue_sizes}")
        if min(queue_sizes) > 0:
            logger.warning(f"[Predict] - Skip processing current frames. Cached Frame: {queue_sizes}")
            return True
        return False


class DisplayProcess(Process):
    def __init__(self, 
                 config: Config, 
                 display_queue: Queue,
                 stop_event,
                 display_3d: bool = True,
                 display_2d: bool = True,
                 display_scale: float = 0.5):
        super().__init__()
        self.config = config
        self.display_queue = display_queue
        self.stop_event = stop_event
        self.display_3d = display_3d
        self.display_2d = display_2d
        self.display_scale = display_scale
            
    def run(self):
        self.visualizer_3d = Visualizer3D(self.config)
        self.visualizer_3d.initialize_3d(self.config.animal['n_tracks'], len(self.config.animal['bodyparts']))
        
        video_name = 'original_composite.mp4' if self.display_3d and self.display_2d else 'original_3d.mp4'
        video_combined_path = Path(self.config.sub_directory['videos_labeled_3d']) / video_name
        self.video_writer = skvideo.io.FFmpegWriter(video_combined_path, inputdict={'-framerate': str(25)}, 
                                                    outputdict={'-vcodec': 'libx264', '-pix_fmt': 'yuv420p', '-preset': 'superfast', '-crf': '23'})
        
        timer = Timer(name='Display', output_path=f'{self.config.project_path}/log/display_latency.npz').start()
        while not self.stop_event.is_set():
            if not self.display_queue.empty():
                data = self.display_queue.get(timeout=5)
                timer.record('Read')

                if self.should_skip_processing():
                    image_3d = None
                    time.sleep(0.001)
                else:
                    image_3d = self.visualizer_3d.get_image_3d(data['points_3d'][:, 0]) if self.display_3d else None
                    timer.record('Visualize 3D')
                    
                if self.display_3d or self.display_2d:
                    skip = all([np.isnan(data[key]).all() for key in ['points_2d', 'points_3d', 'bboxes']])
                    self.display_images(data['images_2d'], image_3d, skip=skip)
                    timer.record('Display')
                timer.show()
            else:
                time.sleep(0.001) # Avoid busy waiting
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()
        timer.show_avg(exclude=['Read'])
        
        cv2.destroyAllWindows()
        logger.info(f'Labeled 3d video saved at {video_combined_path}')
        logger.info('Display Process Finished')

    def should_skip_processing(self) -> bool:
        """Decides whether to skip processing based on queue size."""
        queue_size = self.display_queue.qsize()
        logger.debug(f"[Display] - Cached Frame: {queue_size}")
        if queue_size > 1:
            logger.warning(f"[Display] - Skip processing current frame. Cached Frame: {queue_size}")
            return True
        return False
    
    def display_images(self, images_2d: np.ndarray, image_3d: np.ndarray, skip: bool = False):
        """Display 2D and/or 3D images with pre-defined layout.
        
        Args:
            images_2d: a list of 2D images to display
            image_3d: a 3D image to display
            skip: a flag to skip displaying
        """
        if image_3d is None:
            image_3d = np.full((1080, 960, 3), 255, dtype=np.uint8)
            
        if self.display_2d and self.display_3d:
            image_store = self.visualizer_3d.combine_images(images_2d, image_3d, scale=self.display_scale)
        elif self.display_2d and not self.display_3d:
            n_images = len(images_2d)
            height, width, _ = images_2d[0].shape
            new_height, new_width = int(height*self.display_scale*2), int(width*self.display_scale*2)
            if len(images_2d) == 1:
                image_store = cv2.resize(images_2d[0], (new_width, new_height))
            else:
                images_2d = [cv2.resize(image, (new_width, new_height)) for image in images_2d]
                n_row, n_col = (n_images+1)//2, 2
                image_store = np.full((new_height*n_row, new_width*n_col, 3), 255, dtype=np.uint8)
                for i in range(n_images):
                    r, c = i // 2, i % 2
                    image_store[r*new_height:(r+1)*new_height, c*(new_width):new_width+c*(new_width)] = images_2d[i]
        elif not self.display_2d and self.display_3d:
            image_store = cv2.resize(image_3d, (int(image_3d.shape[1]*self.display_scale), int(image_3d.shape[0]*self.display_scale)))

        if not skip:
            image_display = cv2.cvtColor(image_store, cv2.COLOR_RGB2BGR)
            cv2.imshow("Real-Time 3D Pose Tracking (Press 'Q' to Exit)", image_display)

        self.video_writer.writeFrame(image_store)