import os
import cv2
from glob import glob
from typing import List, Tuple, Dict, Any

from marmopose.utils.common import get_cam_name
from marmopose.utils.io import save_coordinates


def capture_event(event: int, x: int, y: int, flags: int, cam_coordinates: List[Tuple[int, int]]) -> None:
    """
    Mouse callback function to add coordinates to the list when left mouse button is clicked.

    Args:
        event: Type of mouse event.
        x: X-coordinate of mouse event.
        y: Y-coordinate of mouse event.
        flags: Any relevant flags related to the mouse event.
        cam_coordinates: List to which coordinates are appended when left mouse button is clicked.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        print((x, y))
        cam_coordinates.append((x, y))


def capture_coordinates(cap: cv2.VideoCapture, cam_name: str) -> List[Tuple[int, int]]:
    """
    Display video frame and capture coordinates of mouse clicks on it.

    Args:
        cap: VideoCapture object from which frames are read.
        cam_name: Name of the camera for which coordinates are being captured.

    Returns:
        List of coordinates captured from the video frame.
    """
    print(f'Setting axes for cam{cam_name}...')
    ret, img = cap.read()
    if not ret:
        return []

    cv2.imshow(f'cam{cam_name}', img)
    cam_coords = []
    cv2.setMouseCallback(f'cam{cam_name}', capture_event, cam_coords)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return cam_coords


def set_coordinates(config: Dict[str, Any], obj_name: str, offset: Tuple[float, float], frame_idx: int = 0) -> None:
    """
    Set coordinates for each camera by capturing from video frames.

    Args:
        config: Configuration dictionary containing project settings.
        obj_name: Name of the object for which coordinates are being set.
        offset: Offset values.
        frame_idx: Frame index from which to capture the coordinates. Defaults to 0.
    """
    project_dir = config['project_dir']
    videos_raw_dir = config['directory']['videos_raw']
    video_extension = config['video_extension']
    calibration_dir = config['directory']['calibration']
    cam_regex = config['cam_regex']

    videos = sorted(glob(os.path.join(project_dir, videos_raw_dir, f'*.{video_extension}')))

    coords = {'offset': offset}
    for video_path in videos:
        cam_name = get_cam_name(video_path, cam_regex)
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        coords[cam_name] = capture_coordinates(cap, cam_name)
        cap.release()

    output_filepath = os.path.join(project_dir, calibration_dir, 'coordinates.toml')
    save_coordinates(output_filepath, obj_name, coords)
    print(f'Coordinates of {obj_name} saved in: {output_filepath}')


# def set_coordinates(config, obj_name, offset, frame_idx=0):
#     """Set object coorinates for each camera."""
#     project_dir = config['project_dir']
#     calibration_dir = config['directory']['calibration']
#     videos_raw_dir = config['directory']['videos_raw']
#     video_extension = config['video_extension']
#     cam_regex = config['cam_regex']

#     videos = sorted(glob(os.path.join(project_dir, videos_raw_dir, '*.'+video_extension)))

#     coords = dict()
#     coords['offset'] = offset
#     for video_fname in videos:
#         cam_name = get_cam_name(video_fname, cam_regex)
#         cap = cv2.VideoCapture(video_fname)
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#         ret, img = cap.read()
#         if ret:
#             print(f'Set axes for cam{cam_name}')
#             cv2.imshow('image', img)

#             cam_coords = []
#             cv2.setMouseCallback('image', capture_event, cam_coords)
#             coords[cam_name] = cam_coords

#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
#         cap.release()
    
#     output_fname = os.path.join(project_dir, calibration_dir, 'coordinates.toml')
#     save_coordinates(output_fname, obj_name, coords)
#     print(f'Coordinates of {obj_name} saved in: {output_fname}')
