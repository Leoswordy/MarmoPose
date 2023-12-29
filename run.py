import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["NUMBA_NUM_THREADS"] = "4"

from marmopose.version import __version__ as marmopose_version
from marmopose.config import load_config
from marmopose.utils.coordinates_setter import set_coordinates
from marmopose.calibration.calibration import calibrate
from marmopose.processing.prediction import predict
from marmopose.processing.triangulation import triangulate
from marmopose.visualization.display_2d import generate_video_2d
from marmopose.visualization.display_3d import generate_video_3d
from marmopose.visualization.display_combined import generate_video_combined


if __name__ == '__main__':
    config_path = 'configs/double.yaml'
    # config = load_config(config_path)
    config = load_config(config_path, 
                         project_dir='D:\ccq\MarmoSync\demos\double', 
                         model_dir=['D:\ccq\MarmoSync\data\models\centroid_v1.9',
                                    'D:\ccq\MarmoSync\data\models\id_centered_instance_v2.4'],
                         vae_path='D:\ccq\MarmoSync\data\models\VAE_v1.h5')

    # Define customized coordinates
    # set_coordinates(config, obj_name='axes', offset=(0, 0, 0), frame_idx=100)

    # Camera calibration
    # calibrate(config, verbose=True)

    # Predict 2D body parts
    predict(config, batch_size=4, verbose=True)

    # 2D Visualization
    # generate_video_2d(config, points_2d_source='original', verbose=True)

    # Compute 3D poses, fill in missing data, optimize coordinates
    triangulate(config, points_2d_source='original', verbose=True)

    # 3D Visualization
    # generate_video_3d(config, points_3d_source='optimized', fps=25, verbose=True)

    # Combine 2D and 3D videos
    # generate_video_combined(config, points_3d_source='optimized', points_2d_source='original', verbose=True)
