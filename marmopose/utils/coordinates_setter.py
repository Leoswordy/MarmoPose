import cv2
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any


def capture_event(event: int, x: int, y: int, flags: int, params: Tuple[List[Tuple[int, int]], int]) -> None:
    """
    Mouse callback function to add coordinates to the list when the left mouse button is clicked.

    Args:
        event: Type of mouse event.
        x: X-coordinate of mouse event.
        y: Y-coordinate of mouse event.
        flags: Any relevant flags related to the mouse event.
        params: Tuple containing list to which coordinates are appended and current point index.
    """
    cam_coordinates, current_point_idx = params
    if event == cv2.EVENT_LBUTTONDOWN:
        point_types = ['original point', 'x-axis point', 'y-axis point']
        print(f'{point_types[current_point_idx[0]]}: ({x}, {y})')
        cam_coordinates.append((x, y))
        current_point_idx[0] += 1


def capture_coordinates(cap: cv2.VideoCapture, cam_name: str) -> List[Tuple[int, int]]:
    """
    Display video frame and capture coordinates of mouse clicks on it.

    Args:
        cap: VideoCapture object from which frames are read.
        cam_name: Name of the camera for which coordinates are being captured.

    Returns:
        List of coordinates captured from the video frame.
    """
    print(f'\nSetting axes for {cam_name}...')
    ret, img = cap.read()
    if not ret:
        return []

    cv2.imshow(cam_name, img)
    cam_coords = []
    current_point_idx = [0]
    cv2.setMouseCallback(cam_name, capture_event, (cam_coords, current_point_idx))

    while len(cam_coords) < 3:
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    return cam_coords


def set_coordinates(config: Dict[str, Any], video_inds: List, obj_name: str, offset: Tuple[float, float, float], frame_idx: int = 0) -> None:
    """
    Set coordinates for each camera by capturing from video frames.

    Args:
        config: Configuration dictionary containing project settings.
        video_inds: The index of videos for setting coordinates.
        obj_name: Name of the object for which coordinates are being set.
        offset: 3D Offset values (x, y, z).
        frame_idx: Frame index from which to capture the coordinates. Defaults to 0.
    """
    project_dir = Path(config['directory']['project'])
    videos_raw_dir = project_dir / config['directory']['videos_raw']
    calibration_dir = project_dir / config['directory']['calibration']

    video_paths = sorted(videos_raw_dir.glob(f"*.{config['video_extension']}"))

    coordinates_dict = {'offset': offset}
    for i, video_path in enumerate(video_paths):
        if i+1 not in video_inds:
            continue
        cam_name = video_path.stem
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        coordinates_dict[cam_name] = capture_coordinates(cap, cam_name)
        cap.release()

    output_path = project_dir / calibration_dir / f'{obj_name}.json'
    with open(output_path, 'w') as f:
        json.dump(coordinates_dict, f, indent=4)

    print(f'Coordinates of {obj_name} saved in: {output_path}')
