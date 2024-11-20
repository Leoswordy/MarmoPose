import logging
from pathlib import Path
from multiprocessing import Queue, Event
from typing import List

from marmopose.config import Config
from marmopose.utils.constants import IP_DICT
from marmopose.realtime.data_processor import PredictProcess
from marmopose.realtime.controller import EventControl

logger = logging.getLogger(__name__)


def realtime_inference(config: Config, 
                       camera_paths: List[str], 
                       display_3d: bool = True,
                       display_2d: bool = False, 
                       display_scale: float = 1,
                       local_video_path: str = None):
    if local_video_path:
        camera_paths = sorted(Path(local_video_path).glob(f"*.mp4"))
        camera_names = [cam_path.stem for cam_path in camera_paths]
        simulate_live = True
    else:
        camera_names = [IP_DICT.get(camera_path, None) for camera_path in camera_paths]

    if None in camera_names or len(camera_names) < 2:
        logger.warning(f"Cameras hasn't been calibrated. Run in 2D mode.")
        display_3d, display_2d = False, False
        
    data_queue = Queue()
    stop_event = Event()

    predict_process = PredictProcess(config=config, 
                                     camera_paths=camera_paths,
                                     camera_names=camera_names,
                                     data_queue=data_queue,
                                     stop_event=stop_event,
                                     display_2d=display_2d,
                                     simulate_live=simulate_live)
    predict_process.start()

    event_control = EventControl(config=config, 
                                 data_queue=data_queue,
                                 stop_event=stop_event, 
                                 display_3d=display_3d, 
                                 display_2d=display_2d,
                                 display_scale=display_scale)
    event_control.run()

    predict_process.join()
    logger.info('Main process finished')

