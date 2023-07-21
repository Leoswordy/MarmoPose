import cv2
import time
import numpy as np
import queue
from threading import Thread
from multiprocessing import Process, Queue, Event
from typing import List, Tuple, Any, Dict

from mayavi import mlab
mlab.options.offscreen = True

import sleap
from marmopose.utils.common import Timer
from marmopose.triangulate import calculate_3d_axes, transform_coordinates, triangulate_without_optimization, triangulate_with_optimization
from marmopose.visualize_3d import initialize_3d, get_frame_3d


class VideoStreamThread(Thread):
    """
    A thread-safe class for capturing video frames from a given camera path.

    Attributes:
        cap: Object for capturing video from a file or camera.
        params: Dict to store the video parameters.
        frame_queue: Queue for storing captured frames.
        stop_flag: Flag to stop the video capturing.
    """

    def __init__(self, camera_path: str, max_queue_size: int = 1500):
        """
        Initialize a new VideoStreamThread.

        Args:
            cam_path: Path to the camera or video file.
            max_queue_size: Maximum size of the frame queue. Defaults to 1500.
        """
        super().__init__()

        self.cap = cv2.VideoCapture(camera_path)
        self.params = {
            'camera_path': camera_path,
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self.cap.get(cv2.CAP_PROP_FPS))
        }

        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.stop_flag = False

    def run(self):
        """
        Run the video capture thread, captures video frames and stores them in the queue.
        """
        # Simulate actual video stream in real time
        frame_interval = 1.0 / self.params.get('fps', 1)

        while not self.stop_flag:
            start_time = time.time()

            ret, frame = self.cap.read()
            if ret: self.frame_queue.put(frame)

            time_elapsed = time.time() - start_time
            if time_elapsed < frame_interval:
                time.sleep(frame_interval - time_elapsed)

        self.cap.release()
        print(f"{self.params['camera_path']} Released!")
    
    def stop(self):
        """
        Stop current thread by setting the flag.
        """
        self.stop_flag = True


class PredictProcess(Process):
    def __init__(self, config: Dict[str, any], camera_paths: List[str], camera_group: Any, 
                 points_3d_queue: Queue, points_2d_queue: Queue, images_2d_queue: Queue, stop_event: Event, 
                 crop_size: int = 600, compute_3d: bool = True, verbose: bool = True):
        super().__init__()
        self.config = config
        self.n_tracks= config['n_tracks']

        self.camera_paths = camera_paths
        self.n_cams = len(camera_paths)
        self.camera_group = camera_group

        self.compute_3d = compute_3d
        if self.compute_3d:
            _, self.rotation_matrix, self.center = calculate_3d_axes(config, camera_group)

        self.points_3d_queue = points_3d_queue
        self.points_2d_queue = points_2d_queue
        self.images_2d_queue = images_2d_queue

        self.crop_size = crop_size
        self.previous_roi = None

        self.stop_event = stop_event

        self.verbose = verbose

    def run(self):
        self.model = sleap.load_model(self.config['model_dir'], progress_reporting='none')
        self.camera_threads = [VideoStreamThread(camera_path) for camera_path in self.camera_paths]
        for camera_thread in self.camera_threads:
            camera_thread.start()
        self.frame_width = self.camera_threads[0].params['width']
        self.frame_height = self.camera_threads[0].params['height']

        timer = Timer().start()
        while not self.stop_event.is_set():
            # if self.verbose: print(f'Cached frames: {self.camera_threads[0].frame_queue.qsize()} | points_2d: {self.points_2d_queue.qsize()} | points_3d: {self.points_3d_queue.qsize()} | images_2d: {self.images_2d_queue.qsize()}')
            frames = self.get_latest_frames()
            timer.record('Read')

            preprocessed_frames = self.preprocess_frames(frames)
            all_points_2d, all_scores_2d = self.perform_inference(preprocessed_frames)
            timer.record('Predict')

            annotated_frames, all_points_shifted = self.postprocess_frames_and_points(frames, all_points_2d)
            self.points_2d_queue.put((all_points_shifted, all_scores_2d))
            # ******************************************
            # Must resize for faster inter-multiprocess communication!
            # Otherwise serialize and deserialize data between processes could be shockingly slow!
            # If you don't need to display the 2D image, it won't matter.
            # TODO: We should find a better way to organize the code.
            resize_shape = (int(self.frame_width/self.n_cams), int(self.frame_height/self.n_cams))
            annotated_frames = [cv2.resize(image, resize_shape) for image in annotated_frames]
            self.images_2d_queue.put(annotated_frames)
            # ******************************************

            if self.compute_3d: 
                all_points_3d, all_scores_3d = self.triangulate(all_points_shifted, all_scores_2d)
            else:
                all_points_3d, all_scores_3d = None, None
            timer.record('Triangulate')
            self.points_3d_queue.put((all_points_3d, all_scores_3d))

            self.previous_roi = self.calculate_roi(all_points_shifted[0]) if self.crop_size is not None else None
            if self.verbose: timer.show()
        
        if self.verbose: timer.show_avg(2)
        for camera_thread in self.camera_threads:
            camera_thread.stop()
            camera_thread.join()
        print('Predict Process Finished!')
        
    def get_latest_frames(self) -> List[np.ndarray]:
        """
        Fetch the latest frames from all cameras.
        
        Returns:
            A list of numpy arrays, each representing the latest frame from each camera.
        """
        return [camera_thread.frame_queue.get() for camera_thread in self.camera_threads]

    def preprocess_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Preprocess the frames before model inference.

        Args:
            frames: A list of frames with size of (num_cameras, width, height, channel).

        Returns:
            A numpy array of preprocessed frames.
        """
        if self.previous_roi is None:
            roi_frames = frames
        else:
            roi_frames = [frame[y:y+h, x:x+w] for frame, (x, y, w, h) in zip(frames, self.previous_roi)]

        preprocessed = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in roi_frames])
        return preprocessed
    
    def perform_inference(self, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs model inference on the frames to get points and scores.
        
        Args:
            frames: Preprocessed frames with shape (n_cams, cropped_width, cropped_height, channels).

        Returns:
            A tuple containing inferred points and scores.
                - all_points_2d: Inferred points with shape (n_tracks, n_cams x batch_size, n_bodyparts, 2).
                - all_scores_2d: Corresponding scores with shape (n_tracks, n_cams x batch_size, n_bodyparts).
        
        Notes:
            The 'predictions' dictionary has the following structure:
                - 'instance_peaks': (n_frames, n_tracks, n_bodyparts, 2) (float32)
                - 'instance_peak_vals': (n_frames, n_tracks, n_bodyparts) (float32)
                - 'instance_scores': (n_frames, n_tracks) (float32)
                - 'centroids': (n_frames, n_tracks, 2) (float32)
                - 'centroid_vals': (n_frames, n_tracks) (float32)
                - 'n_valid': (n_frames,) (int32)
        """
        predictions = self.model.inference_model.predict_on_batch(frames, numpy=True)
        all_points_2d = np.swapaxes(predictions['instance_peaks'], 0, 1)
        all_scores_2d = np.swapaxes(predictions['instance_peak_vals'], 0, 1)
        return all_points_2d, all_scores_2d
    
    def triangulate(self, all_points: np.ndarray, all_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes 3D points and scores using triangulation.

        Args:
            all_points: Inferred points with shape (n_tracks, n_cams x batch_size, n_bodyparts, 2).
            all_scores: Corresponding scores with shape (n_tracks, n_cams x batch_size, n_bodyparts).

        Returns:
            A tuple containing 3D points and scores.
                - all_points_3d: 3D points with shape (n_tracks, n_frames=1, n_bodyparts, 3).
                - all_scores_3d: Corresponding 3D scores with shape (n_tracks, n_frames=1, n_bodyparts).
        """
        _, _, n_bodyparts, _ = all_points.shape
        all_points_3d, all_scores_3d = [], []

        for points, scores in zip(all_points, all_scores):
            points = points.reshape(self.n_cams, -1, n_bodyparts, 2)
            scores = scores.reshape(self.n_cams, -1, n_bodyparts)
            if self.config['triangulation']['optim']:
                points_3d, scores_3d, _, _ = triangulate_with_optimization(self.config, self.config['bodyparts'], points, scores, self.camera_group, verbose=False)
            else:
                points_3d, scores_3d, _, _ = triangulate_without_optimization(points, scores, self.camera_group, verbose=False)
            points_3d = transform_coordinates(points_3d, self.rotation_matrix, self.center)
            all_points_3d.append(points_3d)
            all_scores_3d.append(scores_3d)

        return np.array(all_points_3d), np.array(all_scores_3d)
    
    def postprocess_frames_and_points(self, frames: List[np.ndarray], all_points: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Annotate frames with keypoints and bounding boxes (if available), and adjust keypoints positions based on bounding boxes.

        Args:
            frames: A list of frames. Each frame is a numpy array of shape (width, height, channel).
            all_points: A list of keypoints. Each item is a numpy array of shape (n_tracks, n_cams x batch_size, n_bodyparts, 2).

        Returns:
            A tuple of two lists: annotated frames and adjusted keypoints.
        """
        # If bounding boxes are available from the previous step
        if self.previous_roi is not None:
            for i in range(len(all_points)):
                all_points[i] = [np.array([[px+x, py+y] for px, py in points]) for points, (x, y, _, _) in zip(all_points[i], self.previous_roi)]
            frames = [cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 4) for frame, (x, y, w, h) in zip(frames, self.previous_roi)]

        for frame, points_per_frame in zip(frames, np.transpose(all_points, (1, 0, 2, 3))):
            valid_points = points_per_frame[~np.isnan(points_per_frame).any(axis=-1)].astype(int)
            for pt in valid_points:
                cv2.circle(frame, tuple(pt), 6, (0, 255, 0), -1)

        return frames, all_points
    
    @staticmethod
    def adjust_positions(min_pos: float, max_pos: float, max_dim: int, crop_size: int) -> Tuple[int, int]:
        """Adjust the position values considering padding and dimension limits.

        Args:
            min_pos: Minimum position (either x or y).
            max_pos: Maximum position (either x or y).
            max_dim: Maximum dimension (either width or height).
            crop_size: Desired size of the cropped image.

        Returns:
            Tuple with adjusted minimum position and corresponding maximum position.
        """
        padding = (crop_size - (max_pos - min_pos)) / 2

        if min_pos < padding:
            min_pos = 0
        elif max_dim - max_pos < padding:
            min_pos = max_dim - crop_size - 1
        else:
            min_pos -= padding

        return int(min_pos)

    def calculate_roi(self, all_points: List[np.ndarray]) -> List[Tuple[int, int, int, int]]:
        """Calculate Region of Interest (ROI) around the detected keypoints.

        Args:
            points: A list of detected keypoints.

        Returns:
            A list of ROI tuples (min_x, min_y, max_x, max_y).
        """
        # Check if a frame has no detected keypoints
        if any(np.isnan(points).all() for points in all_points):
            return None

        roi = []
        for points in all_points:
            min_x, max_x = np.nanmin(points[:, 0]), np.nanmax(points[:, 0])
            min_y, max_y = np.nanmin(points[:, 1]), np.nanmax(points[:, 1])

            min_x = self.adjust_positions(min_x, max_x, self.frame_width, self.crop_size)
            min_y = self.adjust_positions(min_y, max_y, self.frame_height, self.crop_size)

            roi.append((min_x, min_y, self.crop_size, self.crop_size))

        return roi


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
    def __init__(self, config: Dict, display_queue: Queue, stop_event: Event, 
                 display_3d: bool, display_2d: bool, display_scale: float, verbose: bool):
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
            n_tracks = self.config['n_tracks']
            bodyparts = self.config['bodyparts']
            n_bodyparts = len(bodyparts)
            bodypart_dict = dict(zip(bodyparts, range(n_bodyparts)))
            scheme = self.config['visualization']['scheme']
            fig, mlab_points, mlab_lines = initialize_3d(n_tracks, n_bodyparts, scheme, bodypart_dict)

        timer = Timer().start()
        while not self.stop_event.is_set():
            if not self.display_queue.empty():
                all_points_3d, images_2d = self.display_queue.get()
                if self.display_3d:
                    image_3d = get_frame_3d(fig, all_points_3d[:, 0], mlab_points, mlab_lines, scheme, bodypart_dict, show_lines=True)
                    timer.record('Get3DImage')
                else:
                    image_3d = None

                if self.display_3d or self.display_2d:
                    self.display_images(image_3d, images_2d)
                    timer.record('Display')
                    if self.verbose: timer.show()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()
                if self.verbose: timer.show_avg(2)
                # break
        
        if self.display_3d: mlab.close(fig)
        cv2.destroyAllWindows()
        print('Display Process Finished!')

    
    def display_images(self, image_3d: np.ndarray, images_2d: List[np.ndarray]) -> None:
        if self.display_2d:
            n_images = len(images_2d)
            if n_images == 1:
                image_display_2d = images_2d[0]
            else:
                height, width = images_2d[0].shape[:2]
                image_display_2d = np.zeros((height*((n_images+1)//2), width*2, 3), dtype=np.uint8)+255
                for idx in range(n_images):
                    i, j = idx//2, idx%2
                    image_display_2d[i*height:(i+1)*height, j*width:(j+1)*width] = images_2d[idx]
            image_display_2d = cv2.resize(image_display_2d, (int(1920*self.display_scale), int(1080*self.display_scale)))

        if self.display_3d:
            image_display_3d = cv2.cvtColor(image_3d, cv2.COLOR_BGR2RGB)
            image_display_3d = cv2.resize(image_display_3d, (int(1920*self.display_scale), int(1080*self.display_scale)))
        
        if self.display_3d and self.display_2d:
            image_display = cv2.hconcat((image_display_2d, image_display_3d))
        elif self.display_3d:
            image_display = image_display_3d
        elif self.display_2d:
            image_display = image_display_2d
        
        cv2.imshow('Realtime Pose Tracking', image_display)
