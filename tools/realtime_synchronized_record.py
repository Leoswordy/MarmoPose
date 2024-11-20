import logging

from marmopose.utils.constants import IP_DICT
from marmopose.utils.helpers import MultiVideoCapture

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


if __name__ == '__main__':
    camera_paths = [
        'rtsp://admin:abc12345@192.168.1.228:554//Streaming/Channels/101', 
        'rtsp://admin:ABC12345@192.168.1.230:554//Streaming/Channels/101', 
        'rtsp://admin:ABC12345@192.168.1.232:554//Streaming/Channels/101', 
        'rtsp://admin:ABC12345@192.168.1.234:554//Streaming/Channels/101'
    ]
    camera_names = [IP_DICT.get(camera_path, None) for camera_path in camera_paths]


    output_dir = '../demos/realtime_record/videos_raw'

    mvc = MultiVideoCapture(camera_paths, camera_names, do_cache=False, output_dir=output_dir, duration=60)
    mvc.start()
    mvc.join()

    logger.info(f"Real-time video streams saved in {output_dir}")
