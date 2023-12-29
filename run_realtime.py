import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2' # Ignore version incompatibility, not safe?
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["NUMBA_NUM_THREADS"] = "4"
# import numexpr as ne
# ne.set_num_threads(ne.detect_number_of_cores())

from marmopose.config import load_config
from marmopose.realtime.main import realtime_inference


if __name__ == '__main__':
    config_path = 'configs/realtime.yaml'
    config = load_config(config_path)

    camera_paths = [
        'rtsp://admin:abc12345@192.168.1.240:554//Streaming/Channels/101', 
        'rtsp://admin:abc12345@192.168.1.242:554//Streaming/Channels/101', 
        'rtsp://admin:abc12345@192.168.1.244:554//Streaming/Channels/101', 
        'rtsp://admin:abc12345@192.168.1.246:554//Streaming/Channels/101',

        # 'rtsp://admin:abc12345@192.168.1.216:554//Streaming/Channels/101', 
        # 'rtsp://admin:abc12345@192.168.1.218:554//Streaming/Channels/101', 
        # 'rtsp://admin:abc12345@192.168.1.220:554//Streaming/Channels/101', 
        # 'rtsp://admin:abc12345@192.168.1.222:554//Streaming/Channels/101',
        # 'rtsp://admin:abc12345@192.168.1.248:554//Streaming/Channels/101', 
        # 'rtsp://admin:abc12345@192.168.1.250:554//Streaming/Channels/101'
    ]

    # Need sleap version >= 1.3.1
    realtime_inference(config, 
                       camera_paths=camera_paths, 
                       compute_3d=True, 
                       display_3d=True, 
                       display_2d=True, 
                       display_scale=1,
                       crop_size=512, 
                       max_queue_size=7500, 
                       verbose=True, 
                       local_mode=True)
