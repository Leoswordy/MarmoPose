import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class Config:
    DEFAULT_CONFIG = {
        'calibration': {
            'board_type': 'checkerboard',
            'board_square_side_length': 45,
            'fisheye': True
        },
        'visualization': {
            'track_cmap': 'Set2',
            'skeleton_cmap': 'hls'
        },
        'threshold': {
            'bbox': 0.2,
            'iou': 0.8,
            'keypoint': 0.5
        },
        'triangulation': {
            'user_define_axes': True,
            'dae_enable': True
        },
        'optimization': {
            'enable': True,
            'n_deriv_smooth': 1,
            'scale_smooth': 2,
            'scale_length': 2,
            'scale_length_weak': 1
        },
        'sub_directory': {
            'calibration': 'calibration',
            'points_2d': 'points_2d',
            'points_3d': 'points_3d',
            'videos_raw': 'videos_raw',
            'videos_labeled_2d': 'videos_labeled_2d',
            'videos_labeled_3d': 'videos_labeled_3d'
        }
    }

    def __init__(self, config_path: str, **kwargs: Any):
        self.config = self.load_config(config_path)

        self.override_config(self.config, **kwargs)

        self.build_directory()

        self.validate_paths()

    def load_config(self, config_path: str) -> dict:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f'Config file not found: {config_path}')

        with open(config_file, 'r') as file:
            config = yaml.safe_load(file) or {}
        
        self.set_defaults(config, self.DEFAULT_CONFIG)
        return config

    @staticmethod
    def set_defaults(target: dict, defaults: dict) -> None:
        for key, value in defaults.items():
            if key not in target:
                target[key] = value
            elif isinstance(value, dict):
                Config.set_defaults(target[key], value)

    def override_config(self, config: dict, **kwargs: Any) -> None:
        # TODO: Sub-dict might have the same key, handle it
        for key, value in kwargs.items():
            if key in config:
                logger.info(f'Overriding config | {key}: {config[key]} -> {value}')
                config[key] = value
            else:
                for sub_value in config.values():
                    if isinstance(sub_value, dict):
                        self.override_config(sub_value, **{key: value})
    
    def build_directory(self) -> None:
        project_dir = Path(self.config['directory']['project'])
        for key, rel_path in self.config['sub_directory'].items():
            full_path = project_dir / rel_path
            self.config['sub_directory'][key] = str(full_path)
            if rel_path not in ['calibration', 'videos_raw']:
                if not full_path.exists():
                    full_path.mkdir(parents=True, exist_ok=True)
    
    def validate_paths(self) -> None:
        for value in self.config['directory'].values():
            if not Path(value).exists():
                raise FileNotFoundError(f'Directory not found: {value}')

    @property
    def directory(self):
        return self.config['directory']

    @property
    def sub_directory(self):
        return self.config['sub_directory']
    
    @property
    def threshold(self):
        return self.config['threshold']
    
    @property
    def animal(self):
        return self.config['animal']
    
    @property
    def triangulation(self):
        return self.config['triangulation']

    @property
    def visualization(self):
        return self.config['visualization']
    
    @property
    def optimization(self):
        return self.config['optimization']
    
    @property
    def calibration(self):
        return self.config['calibration']

    def __repr__(self) -> str:
        return yaml.dump(self.config)