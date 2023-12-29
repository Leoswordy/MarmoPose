import cv2
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any

from marmopose.calibration.cameras import CameraGroup
from marmopose.calibration.boards import Checkerboard
from marmopose.utils.helpers import orthogonalize_vector
from marmopose.utils.data_io import load_axes


def calibrate(config: Dict[str, Any], verbose: bool = True) -> None:
    project_dir = Path(config['directory']['project'])
    calibration_dir = project_dir / config['directory']['calibration']
    calib_video_paths = sorted(calibration_dir.glob(f"*.{config['video_extension']}"))
    output_dir = project_dir / config['directory']['calibration']
    output_path = output_dir / 'camera_params.json'

    cam_names, video_list = get_video_list(calib_video_paths)
    board = get_calibration_board(config)

    if not output_path.exists():
        detected_file = output_dir / 'detected_boards.pickle'
        if detected_file.exists():
            if verbose: print(f'Loading detected boards from: {detected_file}')
            with open(detected_file, 'rb') as f:
                all_rows = pickle.load(f)
        else:
            if verbose: print('Detecting boards in videos...')
            all_rows = get_rows_videos(video_list, board, verbose)
            with open(detected_file, 'wb') as f:
                pickle.dump(all_rows, f)
        
        cgroup = CameraGroup.from_names(cam_names, config['calibration']['fisheye'])
        cgroup.set_camera_sizes_videos(video_list)

        cgroup.calibrate_rows(all_rows, board, 
                              init_intrinsics=True, init_extrinsics=True, 
                              n_iters=10, start_mu=15, end_mu=1, 
                              max_nfev=200, ftol=1e-4, 
                              n_samp_iter=200, n_samp_full=1000, 
                              error_threshold=2.5, verbose=verbose)
    else:
        if verbose: print(f'Calibration result already exists in: {output_path}')
        cgroup = CameraGroup.load_from_json(str(output_path))

    if config['triangulation']['user_define_axes']:
        update_extrinsics_by_user_define_axes(config, cgroup, verbose)

    cgroup.save_to_json(output_path)
    print(f'Calibration done! Result stored in: {output_path}')


def get_video_list(video_paths):
    cam_videos = defaultdict(list)
    cam_names = set()
    for video_path in video_paths:
        name = video_path.stem
        cam_videos[name].append(str(video_path))
        cam_names.add(name)

    cam_names = sorted(cam_names)
    video_list = [sorted(cam_videos[cname]) for cname in cam_names]
    
    return cam_names, video_list


def get_calibration_board(config: Dict[str, Any]) -> Checkerboard:
    calibration_settings = config['calibration']
    board_size = calibration_settings['board_size']

    return Checkerboard(squaresX=board_size[0], 
                        squaresY=board_size[1], 
                        square_length=calibration_settings['board_square_side_length'])


def get_rows_videos(video_list, board, verbose=True):
    all_rows = []

    for videos in video_list:
        rows_cam = []
        for vnum, vidname in enumerate(videos):
            if verbose: print(vidname)
            rows = board.detect_video(vidname, prefix=vnum, progress=verbose)
            if verbose: print("{} boards detected".format(len(rows)))
            rows_cam.extend(rows)
        all_rows.append(rows_cam)

    return all_rows


def update_extrinsics_by_user_define_axes(config, camera_group, verbose=True):
    project_dir = Path(config['directory']['project'])
    axes_path = project_dir / config['directory']['calibration'] / 'axes.json'
    if not axes_path.exists():
        print(f'Axes file not found in: {axes_path}')
    else:
        if verbose: print('Updating extrinsics by user-defined axes')
        axes = load_axes(axes_path)
        T = construct_transformation_matrix(camera_group, axes)
        for camera in camera_group.cameras:
            update_camera_parameters(camera, T)


def update_camera_parameters(camera, T):
    # Update the rotation vector
    old_R, _ = cv2.Rodrigues(camera.get_rotation())
    new_R = old_R @ T[:3, :3].T 
    new_rvec, _ = cv2.Rodrigues(new_R)
    
    # Update the translation vector
    old_t = camera.get_translation()
    new_t = old_t - old_R @ T[:3, :3].T @ T[:3, 3]

    camera.set_rotation(new_rvec.flatten())
    camera.set_translation(new_t.flatten())


def construct_transformation_matrix(camera_group, axes):
    offset = np.array(axes['offset'])
    axes_2d = np.array([axes[cam_name] for cam_name in camera_group.get_names()], dtype=np.float32)
    # TODO: Triangulate from a subset of cameras
    axes_3d = camera_group.triangulate(axes_2d, undistort=True, verbose=False) - offset

    new_x_axis = axes_3d[1] - axes_3d[0]
    new_y_axis = orthogonalize_vector(axes_3d[2] - axes_3d[0], new_x_axis)
    new_z_axis = np.cross(new_x_axis, new_y_axis)
    
    R = np.vstack([new_x_axis, new_y_axis, new_z_axis])
    R /= np.linalg.norm(R, axis=1)[:, None]
    
    # Construct transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ axes_3d[0]

    return T