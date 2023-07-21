import os
from glob import glob
from multiprocessing import Queue, Event
from typing import List, Dict

from marmopose.realtime.processor import PredictProcess
from marmopose.realtime.control import EventControl
from marmopose.utils.common import get_camera_group, get_cam_name


IP_DICT = {
    'rtsp://admin:abc12345@192.168.1.240:554//Streaming/Channels/101': '1', 
    'rtsp://admin:abc12345@192.168.1.242:554//Streaming/Channels/101': '2', 
    'rtsp://admin:abc12345@192.168.1.244:554//Streaming/Channels/101': '3', 
    'rtsp://admin:abc12345@192.168.1.246:554//Streaming/Channels/101': '4', 
    'rtsp://admin:abc12345@192.168.1.248:554//Streaming/Channels/101': '5', 
    'rtsp://admin:abc12345@192.168.1.250:554//Streaming/Channels/101': '6'
}


def realtime_inference(config: Dict, 
                       camera_paths: List[str], 
                       compute_3d: bool = True, 
                       display_3d: bool = True,
                       display_2d: bool = False, 
                       display_scale: float = 0.5, 
                       crop_size: int = 600, 
                       max_queue_size: int = 1500, 
                       verbose: bool = True, 
                       local_mode: bool = False):
    """
    Performs real-time inference and display results. 

    Args:
        config: Configuration dictionary.
        camera_paths: Paths to the camera. Typically IP addresses.
        compute_3d: Compute 3D predictions. Defaults to True.
        display_3d: Display 3D predictions. Defaults to True.
        display_2d: Display 2D predictions. Defaults to False.
        display_scale: Display scaling factor. Defaults to 0.5.
        crop_size: Crop size for the predictions. Defaults to 600.
        max_queue_size: Maximum queue size for multiprocessing queues. Defaults to 1500.
        verbose: Print verbose logs. Defaults to True.
        local_mode: If true, use local videos for debugging. Defaults to False.
    """
    if local_mode:
        project_dir = config['project_dir']
        videos_raw_dir = config['directory']['videos_raw']
        video_extension = config['video_extension']
        cam_regex = config['cam_regex']
        camera_paths = sorted(glob(os.path.join(project_dir, videos_raw_dir, '*.' + video_extension)))[:len(camera_paths)]
        camera_names = [get_cam_name(camera_path, cam_regex) for camera_path in camera_paths]
    else:
        camera_names = [IP_DICT.get(camera_path, None) for camera_path in camera_paths]

    if None in camera_names or len(camera_names) < 2:
        print(f"Cameras hasn't been calibrated. Run in 2D mode.")
        compute_3d, display_3d, display_2d = False, False, True
        camera_group = None
    else:
        camera_group = get_camera_group(config, camera_names)
        
    points_3d_queue = Queue(maxsize=max_queue_size)
    points_2d_queue = Queue(maxsize=max_queue_size)
    images_2d_queue = Queue(maxsize=max_queue_size)
    stop_event = Event()

    predict_process = PredictProcess(config=config, 
                                     camera_paths=camera_paths, 
                                     camera_group=camera_group, 
                                     points_3d_queue=points_3d_queue, 
                                     points_2d_queue=points_2d_queue, 
                                     images_2d_queue=images_2d_queue, 
                                     stop_event=stop_event, 
                                     crop_size=crop_size, 
                                     compute_3d=compute_3d, 
                                     verbose=verbose)
    predict_process.start()

    event_control = EventControl(config=config, 
                                 points_3d_queue=points_3d_queue, 
                                 points_2d_queue=points_2d_queue,
                                 images_2d_queue=images_2d_queue,
                                 stop_event=stop_event, 
                                 display_3d=display_3d, 
                                 display_2d=display_2d,
                                 display_scale=display_scale,
                                 verbose=verbose)
    event_control.run()
    
    predict_process.join()
    print('Main process finished!')
