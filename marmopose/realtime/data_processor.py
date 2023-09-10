import cv2
import time
import numpy as np
from multiprocessing import Process, Queue, Event
from typing import List, Tuple, Any, Dict
from mayavi import mlab
mlab.options.offscreen = True

from marmopose.utils.helpers import get_color_list, Timer, VideoStreamThread
from marmopose.processing.triangulation import reconstruct_3d_coordinates
from marmopose.processing.optimization import optimize_coordinates
from marmopose.visualization.display_3d import initialize_3d, get_image_3d
from marmopose.visualization.display_combined import combine_images
from marmopose.realtime.predictor import RealtimeSingleInstancePredictor, RealtimeMultiInstacePredictor


class PredictProcess(Process):
    def __init__(self, 
                 config: Dict[str, any], 
                 camera_paths: List[str], 
                 camera_group: Any, 
                 points_3d_queue: Queue, 
                 points_2d_queue: Queue, 
                 images_2d_queue: Queue, 
                 stop_event: Event, 
                 crop_size: int = 600, 
                 compute_3d: bool = True, 
                 verbose: bool = True):
        super().__init__()
        self.config = config
        self.n_tracks= config['animal']['number']

        self.camera_paths = camera_paths
        self.n_cams = len(camera_paths)
        self.camera_group = camera_group

        self.compute_3d = compute_3d

        self.points_3d_queue = points_3d_queue
        self.points_2d_queue = points_2d_queue
        self.images_2d_queue = images_2d_queue
        self.points_3d_cache = [] # For optimization

        self.crop_size = crop_size

        self.stop_event = stop_event

        self.verbose = verbose

    def run(self):
        if self.n_tracks == 1:
            self.predictor = RealtimeSingleInstancePredictor(self.config, n_cams=self.n_cams, crop_size=self.crop_size)
        else:
            self.predictor = RealtimeMultiInstacePredictor(self.config, n_cams=self.n_cams, crop_size=self.crop_size)

        self.camera_threads = [VideoStreamThread(str(path), simulate_live=True, verbose=True) for path in self.camera_paths]
        for camera_thread in self.camera_threads:
            camera_thread.start()
        width, height = self.camera_threads[0].get_param('width'), self.camera_threads[0].get_param('height')

        timer = Timer().start()
        while not self.stop_event.is_set():
            # if self.verbose: print(f'Cached frames: {[cam.get_qsize() for cam in self.camera_threads]} | points_2d: {self.points_2d_queue.qsize()} | points_3d: {self.points_3d_queue.qsize()} | images_2d: {self.images_2d_queue.qsize()}')
            images = self.get_latest_frames()
            timer.record('Read')

            # Image sclae effect the running speed most significantly rather than the model size.!!!!!!!!
            all_points_with_score_2d = self.predictor.predict(images)
            self.points_2d_queue.put(all_points_with_score_2d)
            timer.record('Predict')

            # ******************************************
            # Must resize for faster inter-multiprocess communication!
            # Otherwise serialize and deserialize data between processes could be shockingly slow!
            # If you don't need to display the 2D image, it won't matter.
            # TODO: We should find a better way to organize the code.
            resize_shape = (int(width/4), int(height/4))
            compressed_images = [cv2.resize(image, resize_shape) for image in images]
            self.images_2d_queue.put(compressed_images)
            # ******************************************

            if self.compute_3d: 
                all_points_3d = self.triangulate(all_points_with_score_2d)
            else:
                all_points_3d = None, None
            self.points_3d_queue.put(all_points_3d)
            self.points_3d_cache.append(all_points_3d) # For optimization
            timer.record('Triangulate')

            if self.verbose: timer.show()
        
        if self.verbose: timer.show_avg(5)
        for camera_thread in self.camera_threads:
            camera_thread.stop()
            camera_thread.join()
        print('Predict Process Finished!')
        
    def get_latest_frames(self) -> np.ndarray:
        """Fetch the latest frames from all cameras.
        
        Returns:
            Numpy arrays, each row representing the latest frame from each camera.
        """
        return np.array([camera_thread.read() for camera_thread in self.camera_threads])
    
    def triangulate(self, all_points_with_scores_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computes 3D points and scores using triangulation.

        Args:
            all_points_with_scores_2d: Inferred points with shape (n_cams, n_tracks, n_bodyparts, 3).

        Returns:
            A tuple containing 3D points and scores.
                - all_points_3d: 3D points with shape (n_tracks, n_frames=1, n_bodyparts, 3).
        """
        n_cams, n_tracks, n_bodyparts, _ = all_points_with_scores_2d.shape
        all_points_3d = []

        for track_idx in range(n_tracks):
            # Keep the second dimension as n_frames=1 for compatibility with the triangulation
            points_with_score_2d = all_points_with_scores_2d[:, track_idx:track_idx+1, :, :]
            
            points_3d = reconstruct_3d_coordinates(points_with_score_2d, self.camera_group, verbose=False)
            if self.config['optimization']['enable']:
                previous_points_3d = self.points_3d_cache[-1][track_idx] if len(self.points_3d_cache) > 0 else None
                if previous_points_3d is not None:
                    points_3d_concat = np.concatenate((previous_points_3d, points_3d), axis=0)
                    points_with_score_2d_concat = np.concatenate((points_with_score_2d, points_with_score_2d), axis=1)
                    points_3d_concat_processed = optimize_coordinates(self.config, self.camera_group, points_3d_concat, points_with_score_2d_concat, -1, verbose=False)
                    points_3d = points_3d_concat_processed[-1:]

            all_points_3d.append(points_3d)

        return np.array(all_points_3d)


class DisplayProcess(Process):
    """Class to create a new process for displaying images.

    Attributes:
        config: A dictionary with configuration parameters.
        display_queue: A multiprocessing queue holding 3D keypoints data and 2D images.
        stop_event: A multiprocessing Event to signal termination.
        display_3d: A flag to indicate whether to display 3D images.
        display_2d: A flag to indicate whether to display 2D images.
        display_scale: A scaling factor for the display size.
        verbose: A flag to indicate whether to print verbose messages.
    """
    def __init__(self, 
                 config: Dict[str, Any], 
                 display_queue: Queue, 
                 stop_event: Event, 
                 display_3d: bool = True, 
                 display_2d: bool = True, 
                 display_scale: float = 0.5, 
                 verbose: bool = True):
        super().__init__()
        self.config = config
        self.display_queue = display_queue
        self.stop_event = stop_event
        self.display_3d = display_3d
        self.display_2d = display_2d
        self.display_scale = display_scale
        self.verbose = verbose

    def run(self):
        if self.display_3d:
            n_tracks = self.config['animal']['number']
            bodyparts = self.config['animal']['bodyparts']
            n_bodyparts = len(bodyparts)
            skeleton = self.config['visualization']['skeleton']
            skeleton_indices = [[bodyparts.index(bp) for bp in line] for line in skeleton]
            track_color_list = get_color_list(self.config['visualization']['track_cmap'], n_tracks, cvtInt=False)
            skeleton_color_list = get_color_list(self.config['visualization']['skeleton_cmap'], len(skeleton_indices), cvtInt=False)
            fig, mlab_points, mlab_lines = initialize_3d(n_tracks, n_bodyparts, skeleton_indices, track_color_list, skeleton_color_list)

        while not self.stop_event.is_set():
            if not self.display_queue.empty():
                all_points_3d, images_2d = self.display_queue.get()
                if self.display_3d:
                    image_3d = get_image_3d(fig, all_points_3d[:, 0], mlab_points, mlab_lines, skeleton_indices, show_lines=True)
                else:
                    image_3d = None

                if self.display_3d or self.display_2d:
                    self.display_images(images_2d, image_3d)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()
            
            time.sleep(0.001) # Avoid busy waiting
        
        if self.display_3d: mlab.close(fig)
        cv2.destroyAllWindows()
        print('Display Process Finished!')


    def display_images(self, images_2d: np.ndarray, image_3d: np.ndarray):
        """Display 2D and/or 3D images with pre-defined layout.
        
        Args:
            images_2d: a list of 2D images to display
            image_3d: a 3D image to display
        """
        if self.display_2d and self.display_3d:
            image_display = combine_images(images_2d, image_3d, scale=self.display_scale)
        elif self.display_2d and not self.display_3d:
            n_images = len(images_2d)
            height, width, _ = images_2d[0].shape
            new_height, new_width = int(height*self.display_scale*2), int(width*self.display_scale*2)
            if len(images_2d) == 1:
                image_display = cv2.resize(images_2d[0], (new_width, new_height))
            else:
                images_2d = [cv2.resize(image, (new_width, new_height)) for image in images_2d]
                n_row, n_col = (n_images+1)//2, 2
                image_display = np.full((new_height*n_row, new_width*n_col, 3), 255, dtype=np.uint8)
                for i in range(n_images):
                    r, c = i // 2, i % 2
                    image_display[r*new_height:(r+1)*new_height, c*(new_width):new_width+c*(new_width)] = images_2d[i]

        image_display = cv2.cvtColor(image_display, cv2.COLOR_RGB2BGR)
        cv2.imshow('Realtime Pose Tracking', image_display)

