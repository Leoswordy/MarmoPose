import os
import re
import time
import cv2
import numpy as np
from queue import Queue
from threading import Thread
from collections import defaultdict
from typing import List, Dict, Any, Optional, Union

from marmopose.cameras import CameraGroup


def get_cam_name(filename: str, cam_regex: str) -> Optional[str]:
    """
    Get the camera name by parsing it from the basename of the file using regular expression.

    Args:
        filename: The full path of the file.
        cam_regex: The regular expression to match the camera name in the file base name.

    Returns:
        The camera name if matched; None otherwise.
    """
    match = re.search(cam_regex, os.path.splitext(os.path.basename(filename))[0])
    return match.groups()[0].strip() if match else None
    

def get_camera_group(config: Dict[str, Any], cam_names: List[str]) -> Any:
    """
    Loads camera group from the calibration file and returns a subset camera group with specific camera names.

    Args:
        config: A dictionary containing configurations.
        cam_names: A list of camera names.

    Returns:
        A subset camera group with specific camera names.
    """
    project_dir = config['project_dir']
    calibration_dir = config['directory']['calibration']
    calibration_path = os.path.join(project_dir, calibration_dir, 'calibration.toml')

    cgroup = CameraGroup.load(calibration_path)
    return cgroup.subset_cameras_names(cam_names)


def get_video_params(filepath: str) -> Dict[str, int]:
    """
    Get video parameters.
    
    Args:
        filepath: Path of the video file.

    Returns:
        A dictionary containing the video width, heigth, frames and fps.
    """
    cap = cv2.VideoCapture(filepath)

    params = dict()
    params['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    params['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    params['frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    params['fps'] = int(cap.get(cv2.CAP_PROP_FPS))

    cap.release()
    return params


class Timer:
    """A simple timer class for recording time events.
    
    Attributes:
        events: A dictionary to store the timing records for each event.
        tik: The timestamp of the latest recorded time.
    """
    def __init__(self):
        """Initializes the Timer instance."""
        self.events = defaultdict(list)
        self.tik = 0.0

    def start(self) -> 'Timer':
        """Starts the timer.

        Returns:
            The Timer instance.
        """
        self.tik = time.time()
        return self

    def record(self, event: str) -> None:
        """Records the time for a specific event.

        Args:
            event: The name of the event.
        """
        self.events[event].append(time.time() - self.tik)
        self.start()

    def show(self, event_idx: int = -1) -> None:
        """Shows the time of the specified event index for each event.

        Args:
            event_idx: The index of the event to show. Defaults to -1 (latest event).
        """
        for key, values in self.events.items():
            print(f'{key}: {round(values[event_idx], 5)}', end='  ')
        print('\n')

    def show_avg(self, begin: int = 1) -> None:
        """Shows the average time for each event from a specified index and the total average time.

        Args:
            begin: The start index for the averaging. Defaults to 0.
        """
        print('\nAverage time:')
        total = 0
        for key, values in self.events.items():
            m = np.mean(values[begin:])
            print(f'{key}: {round(m, 5)}', end=' | ')
            total += m
        print(f'Total: {round(total, 5)}')
        print('\n')


class VideoStream(Thread):
    """ 
    A class for video streaming which inherits from Thread.

    Attributes:
        cap: Video capture object.
        params: Parameters of the video.
        grab_interval: Frame grabbing interval.
        queue: A queue to store video frames.
        stop_flag: A flag used to stop the thread.
    """

    def __init__(self, path: str, cache_time: int = 30) -> None:
        """
        Args:
            path: The path to the video file.
            cache_time: Duration of the video to be cached in the queue. Defaults to 30.
        """
        super().__init__()
        self.cap = cv2.VideoCapture(path)
        self.params = {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'frames':  int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': int(self.cap.get(cv2.CAP_PROP_FPS))
        }
        self.grab_interval = 0.1 / self.params['fps']
        self.queue = Queue(maxsize = self.params['fps'] * cache_time)
        self.stop_flag = False

    def run(self) -> None:
        """
        Continuously grabs frames from the video and put them in the queue.
        """
        while not self.stop_flag:
            if not self.queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stop()
                    return
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.queue.put(frame)

    def read(self) -> np.ndarray:
        """
        Reads a frame from the queue.

        Returns:
            The frame read from the queue.
        """
        return self.queue.get()

    def read_batch(self, batch_size: int = 1) -> List[np.ndarray]:
        """
        Reads a batch of frames from the queue.

        Args:
            batch_size: The number of frames to read. Defaults to 1.

        Returns:
            A list of frames.
        """
        return [self.queue.get() for _ in range(batch_size)]

    def stop(self) -> None:
        """
        Stops the video stream.
        """
        self.stop_flag = True
        self.cap.release()
    
    def get_qsize(self) -> int:
        """
        Gets the current size of the queue.

        Returns:
            The current size of the queue.
        """
        return self.queue.qsize()

    def get_params(self) -> Dict[str, Union[int, float]]:
        """
        Gets the parameters of the video.

        Returns:
            The parameters of the video.
        """
        return self.params
