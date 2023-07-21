import os
import toml
from typing import Dict


DEFAULT_CONFIG = {
    'video_extension': 'mp4',
    'calibration': {
        'fisheye': True,
        'init_file': None
    },
    'filter': {
        'enable': False,
        'type': 'myfilter',
        'score_threshold': 0.1
    },
    'triangulation': {
        'optim': False,
        'score_threshold': 0.1,
        'scale_smooth': 2,
        'scale_length': 1,
        'scale_length_weak': 1,
        'reproj_error_threshold': 5,
        'n_deriv_smooth': 1,
        'constraints': [],
        'constraints_weak': []
    },
    'directory': {
        'calibration': 'calibration',
        'poses_2d': 'poses_2d',
        'poses_2d_filtered': 'poses_2d_filtered',
        'poses_3d': 'poses_3d',
        'poses_3d_filtered': 'poses_3d_filtered',
        'videos_raw': 'videos_raw',
        'videos_labeled_2d': 'videos_labeled_2d',
        'videos_labeled_2d_filtered': 'videos_labeled_2d_filtered',
        'videos_labeled_3d': 'videos_labeled_3d',
        'videos_labeled_3d_filtered': 'videos_labeled_3d_filtered'
    }
}


def full_path(path: str) -> str:
    """
    Gets the absolute path given a path, expanding user home directory (~) symbols if present.
    
    Args:
        path: The original path.
        
    Returns:
        The full normalized path.
    """
    return os.path.normpath(os.path.abspath(os.path.expanduser(path)))


def load_config(config_filepath: str) -> Dict[str, any]:
    """
    Loads the configuration file. If the file does not exist, default configurations are created.
    
    Args:
        config_filepath: The name of the configuration file.
        
    Returns:
        The loaded configuration.
    """
    if os.path.exists(config_filepath):
        config = toml.load(config_filepath)
    else:
        config = {'project_dir': os.getcwd()}

    config['project_dir'] = full_path(config['project_dir'])

    for key, value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = value
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                if subkey not in config[key]:
                    config[key][subkey] = subvalue

    return config