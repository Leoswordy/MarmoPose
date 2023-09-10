from pathlib import Path
from multiprocessing import Queue, Event
from typing import List, Dict, Any

from marmopose.realtime.data_processor import PredictProcess
from marmopose.realtime.controller import EventControl
from marmopose.calibration.cameras import CameraGroup
    

def get_camera_group(config: Dict[str, Any], cam_names: List[str]) -> Any:
    """
    Loads camera group from the calibration file and returns a subset camera group with specific camera names.

    Args:
        config: A dictionary containing configurations.
        cam_names: A list of camera names.

    Returns:
        A subset camera group with specific camera names.
    """
    project_dir = Path(config['directory']['project'])
    calibration_path = project_dir / config['directory']['calibration'] / 'camera_params.json'

    camera_group = CameraGroup.load_from_json(calibration_path)
    return camera_group.subset_cameras_names(cam_names)


IP_DICT = {
    'rtsp://admin:abc12345@192.168.1.240:554//Streaming/Channels/101': 'cam1', 
    'rtsp://admin:abc12345@192.168.1.242:554//Streaming/Channels/101': 'cam2', 
    'rtsp://admin:abc12345@192.168.1.244:554//Streaming/Channels/101': 'cam3', 
    'rtsp://admin:abc12345@192.168.1.246:554//Streaming/Channels/101': 'cam4', 
    'rtsp://admin:abc12345@192.168.1.248:554//Streaming/Channels/101': 'cam5', 
    'rtsp://admin:abc12345@192.168.1.250:554//Streaming/Channels/101': 'cam6'
}


def realtime_inference(config: Dict[str, Any], 
                       camera_paths: List[str], 
                       compute_3d: bool = True, 
                       display_3d: bool = True,
                       display_2d: bool = False, 
                       display_scale: float = 0.5, 
                       crop_size: int = 640, 
                       max_queue_size: int = 1500, 
                       verbose: bool = True, 
                       local_mode: bool = False):
    if local_mode:
        project_dir = Path(config['directory']['project'])
        videos_raw_dir = project_dir / config['directory']['videos_raw']
        camera_paths = sorted(videos_raw_dir.glob(f"*.{config['video_extension']}"))
        camera_names = [cam_path.stem for cam_path in camera_paths]
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
