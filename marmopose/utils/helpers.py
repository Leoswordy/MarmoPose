import logging
import time
import threading
from collections import defaultdict
from pathlib import Path
import queue
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import seaborn as sns
import skvideo.io
import av


logger = logging.getLogger(__name__)


def setup_logging(log_file_path: str = None) -> None:
    if log_file_path is None:
        log_file_path = f'{time.strftime("%Y%m%d")}.log'

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file_path, mode='a')

    c_handler.setLevel(logging.DEBUG)
    f_handler.setLevel(logging.DEBUG)

    c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)


def move_log_file(to_dir: str, from_path: str = None) -> None:
    if from_path is None:
        from_path = f'{time.strftime("%Y%m%d")}.log'
    from_path = Path(from_path)
    
    to_dir = Path(to_dir) / 'log'
    if not to_dir.exists():
        to_dir.mkdir(parents=True, exist_ok=True)
    to_path = to_dir / f'{time.strftime("%Y%m%d_%H%M%S")}.log'

    logger.info(f"Log file saved at {to_path}")
    logging.shutdown()

    from_path.rename(to_path)
    

def drain_queue(q):
    try:
        while True:
            q.get_nowait()
    except Exception:
        pass


def get_color_list(cmap_name: str, number: int = None, cvtInt: bool = True) -> List[Tuple]:
    cols = sns.color_palette(cmap_name, number) if number else sns.color_palette(cmap_name)
    if cvtInt:
        cols = [tuple(int(c*255) for c in col) for col in cols]
    return cols


def get_video_params(video_path: Path) -> Dict[str, Union[int, float]]:
    cap = cv2.VideoCapture(str(video_path))
    params = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return params


def orthogonalize_vector(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return u - project_vector(v, u)


def project_vector(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return u * np.dot(v, u) / np.dot(u, u)


class BaseVideoThread(threading.Thread):
    def __init__(self, path, name, frame_queue, stop_event, barrier, do_cache=True, output_dir=None):
        super().__init__()
        self.path = str(path)
        self.name = name

        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.barrier = barrier

        self.do_cache = do_cache

        self.output_path = Path(output_dir) / f"{name}.mp4" if output_dir is not None else None

        self.params = None

    def run(self):
        raise NotImplementedError

    def stop(self):
        self.stop_event.set()

    def get_params(self):
        while self.params is None:
            time.sleep(0.01)
        return self.params


class LocalVideoThread(BaseVideoThread):
    def __init__(self, path, name, frame_queue, stop_event, barrier, do_cache=True, output_dir=None,
                 simulate_live=False, start_frame_idx=0, end_frame_idx=None):
        super().__init__(path, name, frame_queue, stop_event, barrier, do_cache, output_dir)

        self.simulate_live = simulate_live
        self.start_frame_idx = start_frame_idx
        self.end_frame_idx = end_frame_idx

    def run(self):
        self.barrier.wait()
        self.container = av.open(self.path)
        stream = self.container.streams.video[0]
        stream.thread_type = 'AUTO'

        fps = stream.average_rate
        self.params = {
            'path': self.path,
            'name': self.name,
            'width': stream.width,
            'height': stream.height,
            'fps': fps
        }

        if self.start_frame_idx > 0:
            start_pts = int(self.start_frame_idx / fps / stream.time_base)
            self.container.seek(start_pts, any_frame=False, backward=True, stream=stream)

        if self.output_path is not None:
            outputdict = {
                '-r': str(fps),
                '-pix_fmt': 'yuv420p',
                '-vcodec': 'libx264',
                '-crf': '23',
                '-preset': 'superfast',
            }
            self.writer = skvideo.io.FFmpegWriter(str(self.output_path), outputdict=outputdict)

        self.barrier.wait()

        for frame in self.container.decode(video=0):
            if self.stop_event.is_set():
                break

            current_frame_idx = int(frame.pts * stream.time_base * fps)
            if current_frame_idx < self.start_frame_idx:
                continue
            if self.end_frame_idx and current_frame_idx >= self.end_frame_idx:
                break

            start_time = time.time()

            if self.do_cache:
                self.frame_queue.put(frame.to_ndarray(format='bgr24'))

            if self.output_path is not None:
                self.writer.writeFrame(frame.to_ndarray(format='rgb24'))

            if self.simulate_live:
                elapsed_time = time.time() - start_time
                sleep_time = max(0, 1.0 / fps - elapsed_time)
                time.sleep(sleep_time)
            else:
                time.sleep(0.001)

        self.container.close()
        if self.output_path is not None:
            self.writer.close()
            logger.info(f"{self.path} saved at {self.output_path}")


class RTSPVideoThread(BaseVideoThread):
    def __init__(self, path, name, frame_queue, stop_event, barrier,
                 do_cache=True, output_dir=None, duration=None):
        super().__init__(path, name, frame_queue, stop_event, barrier, do_cache, output_dir)
        self.duration = duration

    def run(self):
        options = {
            'rtsp_transport': 'tcp',  # Use TCP for RTSP
            'fflags': 'nobuffer',     # Reduce buffering
            'max_delay': '0',         # Minimize delay
        }

        self.barrier.wait()
        self.container = av.open(self.path, options=options)
        stream = self.container.streams.video[0]
        stream.thread_type = 'AUTO'

        logger.info(f'Stream {self.name} open at {time.time()}')

        fps = stream.average_rate
        self.params = {
            'path': self.path,
            'name': self.name,
            'width': stream.width,
            'height': stream.height,
            'fps': fps
        }

        if self.output_path is not None:
            outputdict = {
                '-r': str(fps),
                '-pix_fmt': 'yuv420p',
                '-vcodec': 'libx264',
                '-crf': '23',
                '-preset': 'superfast',
            }
            self.writer = skvideo.io.FFmpegWriter(str(self.output_path), outputdict=outputdict)

        self.barrier.wait()
        logger.info(f"Stream {self.name} start reading at {time.time()}")

        for frame_idx, frame in enumerate(self.container.decode(video=0)):
            if self.stop_event.is_set():
                break
                
            if self.duration is not None and frame_idx >= int(self.duration * fps):
                break

            pts, timestamp = frame.pts, frame.time
            if frame_idx % int(self.params['fps']*10) == 0:
                logger.info(f"Stream {self.name}: PTS: {pts}, Time: {timestamp}")

            if self.do_cache:
                self.frame_queue.put(frame.to_ndarray(format='bgr24'))

            if self.output_path is not None:
                self.writer.writeFrame(frame.to_ndarray(format='rgb24'))

        self.container.close()
        if self.output_path is not None:
            self.writer.close()
            logger.info(f"{self.path} saved at {self.output_path}")
        
        logger.debug(f"{self.path} Released!")


class MultiVideoCapture:
    def __init__(self, paths, names, do_cache=True, output_dir=None, queue_size=50,
                 simulate_live=False, start_frame_idx=0, end_frame_idx=None, duration=None):
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        self.queues = [queue.Queue(maxsize=queue_size) for _ in paths]
        self.stop_event = threading.Event()
        barrier = threading.Barrier(len(paths))

        self.camera_threads = []
        for path, name, q in zip(paths, names, self.queues):
            if str(path).startswith('rtsp://'):
                thread = RTSPVideoThread(path, name, q, self.stop_event, barrier, 
                                         do_cache, output_dir, duration)
            else:
                thread = LocalVideoThread(path, name, q, self.stop_event, barrier,
                                          do_cache, output_dir, simulate_live, start_frame_idx, end_frame_idx)
            self.camera_threads.append(thread)

    def start(self):
        for t in self.camera_threads:
            t.start()

    def get_next_frames(self, timeout=1.0):
        frames = []
        for q in self.queues:
            try:
                frame = q.get(timeout=timeout)
                frames.append(frame)
            except Exception:
                return None
        return frames
    
    def get_qsizes(self):
        return [q.qsize() for q in self.queues]

    def get_params(self):
        return [t.get_params() for t in self.camera_threads]
    
    def join(self):
        for t in self.camera_threads:
            t.join()

    def stop(self):
        self.stop_event.set()
        self.join()
        

class Timer:
    """A simple timer class for recording time events.
    
    Attributes:
        events: A dictionary to store the timing records for each event.
        tik: The timestamp of the latest recorded time.
        name: The name of the timer.
        output_path: The path to save the timing records.
    """
    def __init__(self, name: str = None, output_path: str = None):
        """Initializes the Timer instance."""
        self.events = defaultdict(list)
        self.tik = 0.0

        self.name = name
        self.output_path = output_path

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
        info = f'[{self.name}] - '
        for key, values in self.events.items():
            info += f'{key}: {round(values[event_idx], 5)} | '
        logger.debug(f'{info}')

    def show_avg(self, exclude: List[str] = None, begin: int = 5) -> None:
        """Shows the average time for each event from a specified index and the total average time.

        Args:
            begin: The start index for the averaging. Defaults to 0.
        """
        logger.info('***** Average time *****')
        total = 0
        info = f'[{self.name}] - '
        for key, values in self.events.items():
            if exclude and key in exclude:
                continue
            m = np.mean(values[begin:])
            info += f'{key}: {round(m, 5)} | '
            total += m
        info += f'Total: {round(total, 5)}\n'
        logger.info(info)

        if self.output_path is not None:
            if not Path(self.output_path).parent.exists():
                Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.output_path, **self.events)
