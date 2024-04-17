import logging
from pathlib import Path

import torch
import numpy as np

from marmopose.config import Config
from marmopose.calibration.cameras import CameraGroup
from marmopose.processing.autoencoder import DaeTrainer
from marmopose.processing.optimization import optimize_coordinates
from marmopose.utils.data_io import load_points_bboxes_2d_h5, save_points_3d_h5

logger = logging.getLogger(__name__)


class Reconstructor3D:
    def __init__(self, config: Config):
        self.init_dir(config)

        self.camera_group = CameraGroup.load_from_json(self.cam_params_path)
        logger.info(f'Loaded camera group from: {self.cam_params_path}')

        self.n_tracks = config.animal['n_tracks']
        self.config = config

        self.build_dae_model(config.directory['dae_model'])
    
    def init_dir(self, config):
        self.cam_params_path = Path(config.sub_directory['calibration']) / 'camera_params.json'
        self.points_2d_path = Path(config.sub_directory['points_2d']) / 'original.h5'
        self.points_3d_path = Path(config.sub_directory['points_3d']) / 'original.h5'
        self.points_3d_optimized_path = Path(config.sub_directory['points_3d']) / 'optimized.h5'
    
    def build_dae_model(self, dae_model_dir):
        checkpoint_files = list(Path(dae_model_dir).glob('*best*.pth'))
        assert len(checkpoint_files) == 1, f'Zero/Multiple best checkpoint files found in {dae_model_dir}'

        dae_checkpoint = str(checkpoint_files[0])
        logger.info(f'Loaded DAE from: {dae_checkpoint}')

        dae = torch.load(dae_checkpoint)
        self.dae_trainer = DaeTrainer(model=dae, bodyparts=self.config.animal['bodyparts'])

    def triangulate(self, all_points_with_score_2d: np.ndarray = None):
        # TODO: Support optimization
        if not all_points_with_score_2d:
            all_points_with_score_2d, _ = load_points_bboxes_2d_h5(self.points_2d_path)
        assert all_points_with_score_2d.shape[1] == self.n_tracks, f'Expected {self.n_tracks} tracks, got {all_points_with_score_2d.shape[1]}'

        for track_idx in range(self.n_tracks):
            track_name = f'track{track_idx+1}'

            points_with_score_2d = all_points_with_score_2d[:, track_idx] # (n_cams, n_frames, n_bodyparts, (x, y, score)))
            points_3d = self.triangulate_instance(points_with_score_2d)
            save_points_3d_h5(points=points_3d, 
                              name=track_name, 
                              file_path=self.points_3d_path)
            
            if self.config.optimization['do_optimize']:
                points_3d_optimized = optimize_coordinates(self.config, self.camera_group, points_3d, points_with_score_2d)
                save_points_3d_h5(points=points_3d_optimized, 
                                  name=track_name, 
                                  file_path=self.points_3d_optimized_path)
    
    def triangulate_instance(self, points_with_score_2d: np.ndarray, verbose: bool = True):
        n_cams, n_frames, n_bodyparts, n_dim = points_with_score_2d.shape
        points_with_score_2d_flat = points_with_score_2d.reshape(n_cams, n_frames*n_bodyparts, n_dim)

        points_3d_flat = self.camera_group.triangulate(points_with_score_2d_flat, undistort=True, verbose=verbose)
        points_3d = points_3d_flat.reshape((n_frames, n_bodyparts, 3)) # (n_frames, n_bodyparts, (x, y, z))

        if self.config.triangulation['dae_enable']:
            points_3d_filled = self.fill_with_dae(points_3d)
            logger.info(f'Filled missing values with denoising autoencoder')
        else:
            points_3d_filled = points_3d

        return points_3d_filled 
    
    def fill_with_dae(self, points_3d: np.ndarray) -> np.ndarray:
        filled_points_3d = points_3d.copy()
        # First find outliers and set them to NaN
        # mask_outliers = np.abs(points_3d - np.nanmean(points_3d, axis=0)) > 2 * np.nanstd(points_3d, axis=0)

        mask_invalid = np.isnan(points_3d)
        res = self.dae_trainer.predict(points_3d)
        filled_points_3d[mask_invalid] = res[mask_invalid]

        return filled_points_3d # (n_frames, n_bodyparts, (x, y, z))