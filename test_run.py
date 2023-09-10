import os
# Ignore version incompatibility, not safe, need to be addressed
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["NUMBA_NUM_THREADS"] = "4"
import numexpr as ne
ne.set_num_threads(ne.detect_number_of_cores())

from marmopose.config import load_config
from marmopose.utils.coordinates_setter import set_coordinates
from marmopose.calibration.calibration import calibrate
from marmopose.processing.prediction import predict
from marmopose.visualization.display_2d import generate_video_2d
from marmopose.processing.triangulation import triangulate
from marmopose.visualization.display_3d import generate_video_3d
from marmopose.visualization.display_combined import generate_video_combined
from marmopose.processing.optimization import parse_constraints
from marmopose.evaluation.metrics_computation import get_reprojected_points_and_error
from marmopose.realtime.main import realtime_inference
from marmopose.version import __version__ as marmopose_version


if __name__ == '__main__':
    config_path = 'configs/single.yaml'
    config = load_config(config_path)
    # config = load_config(config_path,
    #                     project_dir = 'D:\ccq\MarmoSync\demos\single_30s_test',
    #                     model_dir = 'D:\ccq\MarmoSync\data\models\single_v1.2')

    # config_path = 'configs/double.yaml'
    # config = load_config(config_path)
    # config = load_config(config_path, 
    #                      project_dir='D:\ccq\MarmoSync\demos\double_8s_test', 
    #                      model_dir=['D:\ccq\MarmoSync\data\models\centroid_v1.3', 
    #                                 'D:\ccq\MarmoSync\data\models\id_centered_instance_v2.1'])

    # set_coordinates(config, obj_name='axes', offset=(0, 0, 0), frame_idx=25)
    # calibrate(config, verbose=True)

    # predict(config, batch_size=4, verbose=True)
    # generate_video_2d(config, points_2d_source='filtered', verbose=True)

    # triangulate(config, points_2d_source='original', verbose=True)
    # generate_video_3d(config, points_3d_source='optimized', fps=25, verbose=True)

    # get_reprojected_points_and_error(config, points_3d_source='optimized', points_2d_source='filtered', verbose=True)
    # generate_video_2d(config, points_2d_source='reprojected', verbose=True)

    # generate_video_combined(config, points_3d_source='optimized', points_2d_source='reprojected', verbose=True)



    # =========================== Real-time test ===========================
    # config_path = 'configs/realtime.yaml'
    # # config = load_config(config_path)
    # config = load_config(config_path,
    #                      project_dir = 'D:\ccq\MarmoSync\demos\double_40s_test',
    #                      model_dir = ['D:\ccq\MarmoSync\data\models\centroid_v1.4', 
    #                                   'D:\ccq\MarmoSync\data\models\id_centered_instance_v1'])

    # camera_paths = [
    #     'rtsp://admin:abc12345@192.168.1.240:554//Streaming/Channels/101', 
    #     'rtsp://admin:abc12345@192.168.1.242:554//Streaming/Channels/101', 
    #     'rtsp://admin:abc12345@192.168.1.244:554//Streaming/Channels/101', 
    #     'rtsp://admin:abc12345@192.168.1.246:554//Streaming/Channels/101'
    # ]

    # # It should be run on sleap version >= 1.3.1
    # realtime_inference(config, 
    #                    camera_paths=camera_paths, 
    #                    compute_3d=True, 
    #                    display_3d=True, 
    #                    display_2d=True, 
    #                    display_scale=0.8,
    #                    crop_size=640, 
    #                    max_queue_size=1500, 
    #                    verbose=True, 
    #                    local_mode=True)
