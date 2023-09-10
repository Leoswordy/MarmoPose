import cv2
import time
import numpy as np
from collections import defaultdict
from queue import Queue
from pathlib import Path
from threading import Thread
from matplotlib.pyplot import get_cmap
from typing import List, Dict, Union, Tuple


def get_color_list(cmap_name: str, number: int, cvtInt: bool = True) -> List[Tuple]:
    """Gets a list of colors from a colormap.
    
    Args:
        cmap_name: The name of the colormap.
        number: The number of colors to get.
        cvtInt: Whether to convert the color values to integers. Defaults to True.
        
    Returns:
        A list of RGB color tuples.
    """
    cmap = get_cmap(cmap_name)
    if cvtInt:
        return [tuple(int(c) for c in cmap(i % len(cmap.colors), bytes=True)[:3]) for i in range(number)]
    else:
        return [tuple(cmap(i % len(cmap.colors), bytes=False)[:3]) for i in range(number)]


def get_video_params(video_path: Path) -> Dict[str, Union[int, float]]:
    cap = VideoStreamThread(str(video_path))
    params = cap.params
    cap.stop()
    return params


def orthogonalize_vector(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return u - project_vector(v, u)


def project_vector(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return u * np.dot(v, u) / np.dot(u, u)


class VideoStreamThread(Thread):
    def __init__(self, path: str, cache_time: int = 30, simulate_live: bool = False, verbose: bool = True) -> None:
        """
        Args:
            path: The path to the video file.
            cache_time: Duration of the video to be cached in the queue. Defaults to 30.
            simulate_live: Whether to simulate a live stream. Defaults to False.
        """
        super().__init__()
        self.cap = cv2.VideoCapture(path)
        self.params = {
            'path': path,
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self.cap.get(cv2.CAP_PROP_FPS)),
            # TODO: Check what is the result if the video is a live stream.
            'frames':  int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        self.frame_queue = Queue(maxsize = self.params['fps'] * cache_time)

        self.simulate_live = simulate_live
        self.verbose = verbose

        self.stop_flag = False

    def run(self) -> None:
        """Continuously grabs frames from the video and put them in the queue."""
        while not self.stop_flag:
            start_time = time.time()
            if not self.frame_queue.full():
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.frame_queue.put(frame)
            # It is important to sleep here, otherwise the thread will consume too much CPU.
            if self.simulate_live:
                elapsed_time = time.time() - start_time
                sleep_time = max(0, 1.0 / self.params['fps'] - elapsed_time)
                time.sleep(sleep_time)
            else:
                time.sleep(0.01)

        self.cap.release()
        if self.verbose: print(f"{self.params['path']} Released!")

    def read(self) -> np.ndarray:
        """Reads a frame from the queue."""
        return self.frame_queue.get()
    
    def read_latest(self) -> np.ndarray:
        """Reads the most recent frame from the queue, dropping any past frames."""
        last_frame = self.frame_queue.get()
        while not self.frame_queue.empty():
            last_frame = self.frame_queue.get()
        return last_frame

    def stop(self) -> None:
        """Stops the video stream."""
        self.stop_flag = True
    
    def get_qsize(self) -> int:
        """Gets the current size of the queue."""
        return self.frame_queue.qsize()

    def get_param(self, key: str) -> Union[str, int]:
        """Get the parameter of the video.

        Args:
            key: The key of the parameter.

        Returns:
            The parameters of the video.
        """
        return self.params[key]
    


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