import os
import pickle
from glob import glob
from collections import defaultdict
from typing import Dict, Any, Union, Tuple, List, Set

from aniposelib.cameras import CameraGroup
from aniposelib.boards import CharucoBoard, Checkerboard

from marmopose.utils.common import get_cam_name


def get_calibration_board(config: Dict[str, Any]) -> Union[CharucoBoard, Checkerboard]:
    """
    Return calibration board based on config.

    Args:
        config: A dictionary containing the configuration settings.

    Raises:
        ValueError: If the board_type is not 'charuco' or 'checkerboard'.
        
    Returns:
        A calibration board of the specified type.
    """
    calibration_settings = config['calibration']
    board_size = calibration_settings['board_size']
    board_type = calibration_settings['board_type'].lower()

    if board_type == 'charuco':
        return CharucoBoard(squaresX=board_size[0], 
                            squaresY=board_size[1],
                            square_length=calibration_settings['board_square_side_length'],
                            marker_length=calibration_settings['board_marker_length'],
                            marker_bits=calibration_settings['board_marker_bits'],
                            dict_size=calibration_settings['board_marker_dict_number'])
    if board_type == 'checkerboard':
        return Checkerboard(squaresX=board_size[0], 
                            squaresY=board_size[1], 
                            square_length=calibration_settings['board_square_side_length'])

    raise ValueError(f"Board type should be 'charuco' or 'checkerboard', not '{board_type}'")


def get_video_list(config: dict) -> Tuple[Set[str], List[List[str]]]:
    """
    Fetches the list of all the video files in the calibration directory and groups them by camera names.

    Args:
        config: A dictionary containing the configuration settings.

    Returns:
        cam_names: List of sorted camera names.
        video_list: List of sorted video file paths for each camera.
    """
    project_dir = config['project_dir']
    calibration_dir = config['directory']['calibration']
    video_extension = config['video_extension']
    cam_regex = config['cam_regex']

    videos = sorted(glob(os.path.join(project_dir, calibration_dir, f'*.{video_extension}')))

    cam_videos = defaultdict(list)
    cam_names = set()
    for vid in videos:
        name = get_cam_name(vid, cam_regex)
        cam_videos[name].append(vid)
        cam_names.add(name)

    cam_names = sorted(cam_names)
    video_list = [sorted(cam_videos[cname]) for cname in cam_names]
    
    return cam_names, video_list


def get_cgroup_and_error(config: dict, cam_names: Set[str], videos: List[str]) -> Tuple[CameraGroup, bool, bool, Any]:
    """
    Fetches the camera group and error if calibration is already done, else creates a new camera group.

    Args:
        config: Dictionary containing the configuration parameters.
        cam_names: List of camera names.
        videos: List of the videos.

    Returns:
        cgroup: Group of cameras.
        skip_calibration: Flag indicating if calibration should be skipped.
        init_params: Flag indicating if intrinsics and extrinsics should be initialized.
        error: Calibration error if it exists, else None.
    """
    project_dir = config['project_dir']
    calibration_dir = config['directory']['calibration']
    output_filepath = os.path.join(project_dir, calibration_dir, 'calibration.toml')

    skip_calibration = False
    init_params = True
    error = None

    if config['calibration']['init_file'] is not None:
        init_filepath = os.path.join(project_dir, config['calibration']['init_file'])
        print(f'Loading initial calibration parameters from: {init_filepath}')
        cgroup = CameraGroup.load(init_filepath)
        init_params = False
        skip_calibration = len(videos) == 0
    elif os.path.exists(output_filepath):
        print(f'Calibration result already exists in: {output_filepath}')
        cgroup = CameraGroup.load(output_filepath)
        error = cgroup.metadata['error']
        init_params = False
        skip_calibration = True
    else:
        if len(videos) == 0:
            print('No videos or calibration file found, please check')
            return None, False, True, None
        cgroup = CameraGroup.from_names(cam_names, config['calibration']['fisheye'])

    return cgroup, skip_calibration, init_params, error


def calibrate(config: dict, verbose: bool = True) -> None:
    """
    Computes camera parameters using videos with checkerboard.

    Args:
        config: Dictionary containing the configuration parameters.
        verbose: If True, detailed logs are printed, else minimal logs are printed.
    """
    cam_names, video_list = get_video_list(config)

    cgroup, skip_calibration, init_params, error = get_cgroup_and_error(config, cam_names, video_list)
    if cgroup is None:
        return

    output_dir = os.path.join(config['project_dir'], config['directory']['calibration'])
    output_filepath = os.path.join(output_dir, 'calibration.toml')

    board = get_calibration_board(config)

    if not skip_calibration:
        rows_filepath = os.path.join(output_dir, 'detected_boards.pickle')
        if os.path.exists(rows_filepath):
            print(f'Loading detected boards from: {rows_filepath}')
            with open(rows_filepath, 'rb') as f:
                all_rows = pickle.load(f)
        else:
            print('Detecting boards in videos...')
            all_rows = cgroup.get_rows_videos(video_list, board, verbose)
            with open(rows_filepath, 'wb') as f:
                pickle.dump(all_rows, f)

        cgroup.set_camera_sizes_videos(video_list)
        
        error = cgroup.calibrate_rows(all_rows, board, 
                                      init_intrinsics=init_params, init_extrinsics=init_params, 
                                      n_iters=20, start_mu=15, end_mu=1, 
                                      max_nfev=200, ftol=1e-4, 
                                      n_samp_iter=200, n_samp_full=1000, 
                                      error_threshold=1, verbose=verbose)

    if error is not None:
        cgroup.metadata['error'] = float(error)

    cgroup.dump(output_filepath)
    print(f'Calibration done! Result stored in: {output_filepath}')
