import sys
sys.path.append('../')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger()

from marmopose.config import Config
from marmopose.realtime.main import realtime_inference


if __name__ == '__main__':
    config_path = '../configs/realtime.yaml'
    project_dir = '../demos/realtime'

    config = Config(
        config_path=config_path,
        n_tracks=1, 

        project=project_dir,
        det_model='../models/detection_model_deployed',
        pose_model='../models/pose_model_deployed',
        dae_model='../models/dae_model',

        dae_enable=False,
        do_optimize=False
    )


    camera_paths = [
        # Should be a list of target RTSP camera streams. e.g.
        # 'rtsp://admin:abc12345@192.168.1.240:554//Streaming/Channels/101'
    ]

    # If local_video_path is set, local videos will be used to simulate realtime streams for demo and debug purposes.
    # Raw videos, 2D predictions, 3D poses and the 3D video will be saved in the project directory
    realtime_inference(config, 
                       camera_paths=camera_paths, 
                       display_3d=True, 
                       display_2d=False, 
                       display_scale=1,
                       local_video_path='../demos/single/videos_raw')
    
