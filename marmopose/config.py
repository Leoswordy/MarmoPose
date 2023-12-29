import yaml
from pathlib import Path
from typing import Dict, Any


DEFAULT_CONFIG = {
    'video_extension': 'mp4',
    'calibration': {
        'board_type': 'checkerboard',
        'fisheye': True
    },
    'visualization': {
        'track_cmap': 'Set2',
        'skeleton_cmap': 'hls'
    },
    'filter': {
        'threshold': 0.2
    },
    'triangulation': {
        'user_define_axes': True
    },
    'optimization': {
        'enable': True,
        'n_deriv_smooth': 1,
        'scale_smooth': 1,
        'scale_length': 2,
        'scale_length_weak': 1,
        'constraints': [],
        'constraints_weak': []
    },
    'directory': {
        'calibration': 'calibration',
        'points_2d': 'points_2d',
        'points_3d': 'points_3d',
        'videos_raw': 'videos_raw',
        'videos_labeled_2d': 'videos_labeled_2d',
        'videos_labeled_3d': 'videos_labeled_3d'
    }
}


def set_defaults(target: Dict[str, Any], defaults: Dict[str, Any]) -> None:
    """
    Recursively sets default values in the target dictionary.

    Args:
        target: The target dictionary where default values are to be set.
        defaults: The dictionary containing default values.
    """
    for key, value in defaults.items():
        if key not in target:
            target[key] = value
        elif isinstance(value, dict):
            set_defaults(target[key], value)
            

def load_config(config_path: str, project_dir: str = None, model_dir: str = None, vae_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file and set default values if not present.

    Args:
        config_path: Path to the YAML configuration file.
        project_dir (optional): Path to the project directory, overrides the value in the config file. Defaults to None.
        model_dir (optional): Path to the model directory, overrides the value in the config file. Defaults to None.
        vae_path (optional): Path to the VAE model file, overrides the value in the config file. Defaults to None.

    Returns:
        A dictionary containing the configuration values.
    """
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    else:
        raise FileNotFoundError(f'Config file not found: {config_path}')

    set_defaults(config, DEFAULT_CONFIG)

    if project_dir is not None: config['directory']['project'] = project_dir
    if model_dir is not None: config['directory']['model'] = model_dir
    if vae_path is not None: config['directory']['vae'] = vae_path

    if not Path(config['directory']['project']).exists():
        raise FileNotFoundError(f'Project directory not found: {config["directory"]["project"]}')

    return config