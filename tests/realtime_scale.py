import os
import cv2
import time
import threading
import numpy as np
import queue
from glob import glob
from multiprocessing import Process, Queue, Event
from typing import List, Tuple, Any

from mayavi import mlab
mlab.options.offscreen = True

import sleap
from marmotrack.utils.common import get_cam_name, get_camera_group, Timer
from marmotrack.triangulate import calculate_3d_axes, transform_coordinates, triangulate_without_optimization, triangulate_with_optimization
from marmotrack.visualize_3d import initialize_3d, get_frame_3d


class VideoStreamThread(threading.Thread):
    """
    A thread-safe class for capturing video frames from a given camera path.

    Attributes:
        cap: Object for capturing video from a file or camera.
        params: Dict to store the video parameters.
        frame_queue: Queue for storing captured frames.
        stop_event: Flag to stop the video capturing.
    """

    def __init__(self, camera_path: str, scale: float, stop_event: Event, max_queue_size: int = 1000):
        """
        Initialize a new VideoStreamThread.

        Args:
            cam_path: Path to the camera or video file.
            scale: The scale of the image.
            max_queue_size: Maximum size of the frame queue. Defaults to 1000.
        """
        threading.Thread.__init__(self)

        self.cap = cv2.VideoCapture(camera_path)
        self.params = {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self.cap.get(cv2.CAP_PROP_FPS))
        }

        # self.frame_shape = (int(self.params['width'] * scale), int(self.params['height'] * scale))
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = stop_event

    def run(self) -> None:
        """
        Run the video capture thread, captures video frames and stores them in the queue.
        """
        # Simulate actual video stream in real time
        frame_interval = 1.0 / self.params.get('fps', 1)

        while not self.stop_event.is_set():
            start_time = time.time()

            ret, frame = self.cap.read()
            if ret: 
                # frame shape (height, width, channels), cv2.resize shape should be (width, height)
                # frame = cv2.resize(frame, self.frame_shape)
                self.frame_queue.put(frame)

            time_elapsed = time.time() - start_time
            if time_elapsed < frame_interval:
                time.sleep(frame_interval - time_elapsed)
        
        self.cap.release()


class ImageProcessorProcess(Process):
    def __init__(self, config: Any, camera_paths: List[str], cgroup: Any,
                 rotation_matrix: np.ndarray, center: np.ndarray, 
                 points_3d_queue: Queue, images_2d_queue: Queue, stop_event: Event, 
                 scale: float=0.5, crop_size: int = 600, compute_3d: bool = True, draw_2d: bool = False, verbose: bool = True):
        super().__init__()
        self.config = config

        self.camera_paths = camera_paths
        self.cgroup = cgroup
        self.n_cams = len(camera_paths)

        self.n_tracks, self.rotation_matrix, self.center = config['n_tracks'], rotation_matrix, center

        self.points_3d_queue = points_3d_queue
        self.images_2d_queue = images_2d_queue

        self.scale = scale
        self.crop_size = crop_size
        self.previous_roi = None

        self.stop_event = stop_event
        self.draw_2d = draw_2d
        self.compute_3d = compute_3d
        self.verbose = verbose

    def run(self):
        """Execute the thread's main loop, which fetches frames, performs inference, and processes results."""
        self.model = sleap.load_model(self.config['model_dir'], progress_reporting='none')
        self.camera_threads = [VideoStreamThread(camera_path, self.scale, self.stop_event) for camera_path in self.camera_paths]
        for camera_thread in self.camera_threads:
            camera_thread.start()
        # self.frame_width = self.camera_threads[0].frame_shape[0]
        # self.frame_height = self.camera_threads[0].frame_shape[1]
        self.frame_width = 1920
        self.frame_height = 1080

        timer = Timer()
        while not self.stop_event.is_set():
            if self.verbose: print(f'frame: {[self.camera_threads[i].frame_queue.qsize() for i in range(self.n_cams)]} | points_3d: {self.points_3d_queue.qsize()} | images_2d: {self.images_2d_queue.qsize()}')
            timer.start()
            frames = self.get_latest_frames()
            timer.record('Read')

            preprocessed_frames = self.preprocess_frames(frames)
            timer.record('Preprocess')

            all_points, all_scores = self.perform_inference(preprocessed_frames)
            timer.record('Predict')

            annotated_frames, all_points_shifted = self.postprocess_frames_and_points(frames, all_points)
            timer.record('Postprocess')

            if self.compute_3d: 
                all_points_3d, all_scores_3d = self.triangulate(all_points_shifted, all_scores)
                timer.record('Triangulate')
                self.points_3d_queue.put(all_points_3d)
                timer.record('Put3D')

            if self.draw_2d: 
                self.images_2d_queue.put(annotated_frames)
                timer.record('Put2D')

            self.previous_roi = self.calculate_roi(all_points_shifted[0]) if self.n_tracks == 1 else None
            timer.record('ROI')
            if self.verbose: timer.show()
        
        if self.verbose: timer.show_avg(2)
        for camera_thread in self.camera_threads:
            camera_thread.stop_event.set()
        
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
                - all_points: Inferred points with shape (n_tracks, n_cams x batch_size, n_bodyparts, 2).
                - all_scores: Corresponding scores with shape (n_tracks, n_cams x batch_size, n_bodyparts).
        """
        # The 'predictions' dictionary has the following structure:
        # 'instance_peaks': (n_frames, n_tracks, n_bodyparts, 2) (float32)
        # 'instance_peak_vals': (n_frames, n_tracks, n_bodyparts) (float32)
        # 'instance_scores': (n_frames, n_tracks) (float32)
        # 'centroids': (n_frames, n_tracks, 2) (float32)
        # 'centroid_vals': (n_frames, n_tracks) (float32)
        # 'n_valid': (n_frames,) (int32)
        print(np.array(frames).shape)
        predictions = self.model.inference_model.predict_on_batch(frames, numpy=True)
        all_points = np.swapaxes(predictions['instance_peaks'], 0, 1)
        all_scores = np.swapaxes(predictions['instance_peak_vals'], 0, 1)
        return all_points, all_scores
    
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
                points_3d, scores_3d, _, _ = triangulate_with_optimization(self.config, self.config['bodyparts'], points, scores, self.cgroup, verbose=False)
            else:
                points_3d, scores_3d, _, _ = triangulate_without_optimization(points, scores, self.cgroup, verbose=False)
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
            if self.draw_2d:
                frames = [cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 4) for frame, (x, y, w, h) in zip(frames, self.previous_roi)]

        # Annotate frames with keypoints
        if self.draw_2d:
            for frame, points_per_frame in zip(frames, np.transpose(all_points, (1, 0, 2, 3))):
                valid_points = points_per_frame[~np.isnan(points_per_frame).any(axis=-1)].astype(int)
                for pt in valid_points:
                    cv2.circle(frame, tuple(pt), 4, (0, 255, 0), -1)
            # frames = [cv2.resize(frame, (int(self.frame_width * 0.5), int(self.frame_height * 0.5)), interpolation=cv2.INTER_NEAREST) for frame in frames]
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


def display_images(image_3d, images_2d, display_scale: float = 1) -> None:
    if images_2d is not None:
        image_top = cv2.hconcat((images_2d[0], images_2d[1]))    
        image_bottom = cv2.hconcat((images_2d[2], images_2d[3]))
        image_display = cv2.vconcat((image_top, image_bottom))
        height, width = image_display.shape[:2]
        image_display = cv2.resize(image_display, (int(width * display_scale), int(height * display_scale)), interpolation=cv2.INTER_NEAREST)
        # print(image_display.shape)
    if image_3d is not None:
        # white_image = np.zeros(image_3d.shape, dtype='uint8')+255
        # image_right = cv2.vconcat((image_3d, white_image))
        # print(image_3d.shape)
        image_display = cv2.hconcat(((image_display, image_3d)))
    
    if image_3d is not None or images_2d is not None:
        cv2.imshow('Realtime pose tracking', image_display)


def realtime_inference_and_show(config, camera_paths=None, draw_2d=False, compute_3d=True, preprocess_scale=0.7, display_scale=1, crop_size=600, max_queue_size = 1000, verbose=True):
    project_dir = config['project_dir']
    videos_raw_dir = config['directory']['videos_raw']
    video_extension = config['video_extension']

    n_tracks = config['n_tracks']
    bodyparts = config['bodyparts']
    n_bodyparts = len(bodyparts)
    bodypart_dict = dict(zip(bodyparts, range(n_bodyparts)))
    scheme = config['labeling']['scheme']

    if camera_paths is None:
        camera_paths = sorted(glob(os.path.join(project_dir, videos_raw_dir, '*.' + video_extension)))
        cam_names = [get_cam_name(path, config['cam_regex']) for path in camera_paths]
    else: 
        # TODO: get cam_names from the ip address, maybe using a dict
        cam_names = ['1', '2', '3', '4']
    cgroup = get_camera_group(config, cam_names)
    # cgroup.resize_cameras(preprocess_scale)
    
    points_3d_queue = Queue(maxsize=max_queue_size)
    images_2d_queue = Queue(maxsize=max_queue_size)

    rotation_matrix, center = np.eye(3), np.zeros(3)
    if compute_3d: 
        axes_3d, rotation_matrix, center = calculate_3d_axes(config, cam_names, cgroup)
        fig, mlab_points, mlab_lines = initialize_3d(n_tracks, n_bodyparts, scheme, bodypart_dict, axes_3d, 2*preprocess_scale*display_scale)

    stop_event = Event()
    camera_group = ImageProcessorProcess(config, camera_paths, cgroup, rotation_matrix, center, points_3d_queue, images_2d_queue, stop_event, preprocess_scale, crop_size, compute_3d, draw_2d, verbose)
    camera_group.start()

    timer = Timer().start()
    image_3d, images_2d = None, None
    while True:
        if compute_3d and not points_3d_queue.empty():
            all_points_3d = points_3d_queue.get()
            # print(all_points_3d[0, 0])
            timer.record("Read3D")
            image_3d = get_frame_3d(fig, all_points_3d[:, 0], mlab_points, mlab_lines, scheme, bodypart_dict, show_lines=False)
            timer.record('Get3DImage')

            if draw_2d and not images_2d_queue.empty():
                images_2d = images_2d_queue.get()
                timer.record('Read2D')

                display_images(image_3d, images_2d, display_scale)
                timer.record('Display')
                if verbose: timer.show()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            if verbose: timer.show_avg(10)
            break
    
    mlab.close(all=True)
    cv2.destroyAllWindows()
