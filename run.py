import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from marmopose.version import __version__ as marmopose_version
from marmopose.config import Config
from marmopose.calibration.calibration import Calibrator
from marmopose.processing.prediction import Predictor
from marmopose.visualization.display_2d import Visualizer2D
from marmopose.visualization.display_3d import Visualizer3D
from marmopose.processing.triangulation import Reconstructor3D

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    logger.info(f'MarmoPose version: {marmopose_version}')

    config_path = 'configs/default.yaml'

    config = Config(
        config_path=config_path,

        # === The following parameters will override the config file, you can also set them in the config file ===

        # n_tracks=1, 
        # project=r'demos/single',

        # det_model='models/detection_model_deployed',
        # pose_model='models/pose_model_deployed',

        # do_optimize=True 
    )

    # # Run calibration, only need to run once as long as the camera setup doesn't change
    # calibrator = Calibrator(config)
    # calibrator.set_coordinates(video_inds=[1, 2], obj_name='axes', offset=(150, 50, 120))
    # calibrator.calibrate()
    

    # # Run 2D prediction
    predictor = Predictor(config, batch_size=4)
    predictor.predict()


    # # Run 3D triangulation
    reconstructor_3d = Reconstructor3D(config)
    reconstructor_3d.triangulate()


    # # Visualize 2D results if needed
    # visualizer_2d = Visualizer2D(config)
    # visualizer_2d.generate_videos_2d()


    # # Visualize 3D results if needed
    # visualizer_3d = Visualizer3D(config)

    # visualizer_3d.generate_video_3d(source='original')
    # visualizer_3d.generate_video_combined(source='original')

    # visualizer_3d.generate_video_3d(source='optimized')
    # visualizer_3d.generate_video_combined(source='optimized')