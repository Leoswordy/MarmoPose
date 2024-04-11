import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning) # TODO: Remove this in formal release

from marmopose.version import __version__ as marmopose_version
from marmopose.config import Config
from marmopose.calibration.calibration import Calibrator
from marmopose.processing.prediction import Predictor
from marmopose.visualization.display_2d import Visualizer2D
from marmopose.visualization.display_3d import Visualizer3D
from marmopose.processing.triangulation import Reconstructor3D

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')


if __name__ == '__main__':
    config_path = 'configs/default.yaml'

    config = Config(
        config_path=config_path,
        # n_tracks=2, 

        project='demos/single',
        # det_model='models/detection_model_deployed',
        # pose_model='models/pose_model_deployed',

        dae_enable=True,
        do_optimize=True
    )

    # calibrator = Calibrator(config)
    # calibrator.set_coordinates(video_inds=[1, 2], obj_name='axes', offset=(150, 50, 120))
    # calibrator.calibrate()
    

    predictor = Predictor(config, batch_size=4)
    predictor.predict()

    reconstructor_3d = Reconstructor3D(config)
    reconstructor_3d.triangulate()

    visualizer_2d = Visualizer2D(config)
    visualizer_2d.generate_videos_2d()

    visualizer_3d = Visualizer3D(config)
    visualizer_3d.generate_video_3d(source='original', start_frame_idx=0)
    visualizer_3d.generate_video_combined(source='original')

    # visualizer_3d.generate_video_3d(source='optimized')
    # visualizer_3d.generate_video_combined(source='optimized')