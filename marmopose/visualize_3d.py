import os
import skvideo.io
import numpy as np
from glob import glob
from tqdm import trange
from matplotlib.pyplot import get_cmap
from mayavi import mlab
mlab.options.offscreen = True
from typing import List, Tuple, Dict, Any

from marmopose.utils.common import get_video_params
from marmopose.utils.io import load_pose_3d


def draw_line(points: np.ndarray, bodyparts: List[str], 
              bodypart_dict: Dict[str, int], color: Tuple[float, float, float]) -> Any:
    """
    Connects points to form a line in 3D space.

    Args:
        points: Numpy array of 3D coordinates.
        bodyparts: List of body parts to be connected.
        bodypart_dict: Dictionary mapping body part names to indices.
        color: Tuple representing RGB color.

    Returns:
        A line object in 3D space.
    """
    indices = [bodypart_dict[bp] for bp in bodyparts]
    return mlab.plot3d(points[indices, 0], points[indices, 1], points[indices, 2],
                       np.ones(len(indices)), reset_zoom=False,
                       color=color, tube_radius=None, line_width=5)


def draw_all_lines(points: np.ndarray, scheme: List[List[str]], 
                   bodypart_dict: Dict[str, int], cmap_name: str = 'Set2') -> List[Any]:
    """
    Connects all lines for a track.

    Args:
        points: Numpy array of 3D coordinates.
        scheme: List of lists where each inner list contains body parts to be connected.
        bodypart_dict: Dictionary mapping body part names to indices.
        cmap_name: Colormap name. Defaults to 'Set3'.

    Returns:
        List of all line objects in 3D space.
    """
    colormap = get_cmap(cmap_name)
    lines = []
    for idx, bodyparts in enumerate(scheme):
        color = colormap((idx+1) % len(colormap.colors))
        line = draw_line(points, bodyparts, bodypart_dict, color=color[:3])
        lines.append(line)

    return lines


def update_all_lines(lines: List[Any], points: np.ndarray, 
                     scheme: List[List[str]], bodypart_dict: Dict[str, int]) -> None:
    """
    Update all lines.

    Args:
        lines: List of all line objects in 3D space.
        points: Numpy array of 3D coordinates.
        scheme: List of lists where each inner list contains body parts to be connected.
        bodypart_dict: Dictionary mapping body part names to indices.
    """
    for line, bodyparts in zip(lines, scheme):
        indices = [bodypart_dict[bp] for bp in bodyparts]
        line.mlab_source.points = points[indices]


def draw_room_grids(width: int, length: int, height: int, grid_size: int) -> None:
    """
    Draw axes grids in 3D space.

    Args:
        width: Room width.
        length: Room length.
        height: Room height.
        grid_size: Grid size.
    """
    col = (1, 1, 1)
    x, y = np.mgrid[:width:grid_size, :length:grid_size]
    mlab.surf(x, y, np.zeros(x.shape), representation='wireframe', color=col, line_width=0.5)
    z, y = np.mgrid[:height:grid_size, :length:grid_size]
    mlab.surf(np.zeros(y.shape), y, z, representation='wireframe', color=col, line_width=0.5)
    x, z = np.mgrid[:width:grid_size, :height:grid_size]
    mlab.surf(x, np.zeros(z.shape), z, representation='wireframe', color=col, line_width=0.5)


def initialize_3d(n_tracks: int, n_bodyparts: int, scheme: List[List[str]], bodypart_dict: Dict[str, int], 
                  axes: np.ndarray = None, scale: float = 1.0, 
                  room_dimensions: List[int] = [680, 980, 780, 30], cmap_name: str = 'Set3') -> Tuple[Any, List[Any], List[Any]]:
    """
    Initialize 3D figure with points and lines.

    Args:
        n_tracks: Number of tracks.
        n_bodyparts: Number of body parts.
        scheme: List of lists where each inner list contains body parts to be connected.
        bodypart_dict: Dictionary mapping body part names to indices.
        axes: 3D axes coordinates.
        scale: The scale of the figure.
        room_dimensions: Room dimensions [width, length, height, grid_size].
        cmap_name: Name of the colormap.

    Returns:
        Initialized 3D figure, list of point objects, and list of line objects.
    """
    fig = mlab.figure(bgcolor=(1, 1, 1), size=(int(1920*scale), int(1080*scale)))
    fig.scene.anti_aliasing_frames = 8
    mlab.clf()

    # Draw room with grids
    draw_room_grids(*room_dimensions)

    # Draw axes
    scale_factor = np.max(room_dimensions)//50
    if axes is not None:
        mlab.points3d(axes[:, 0], axes[:, 1], axes[:, 2], color=(0.8, 0.8, 0.8), scale_factor=scale_factor)

    # Initialize points and lines
    cmap = get_cmap(cmap_name)
    mlab_points, mlab_lines = [], []
    for track_idx in range(n_tracks):
        points = np.zeros((n_bodyparts, 3))
        color = cmap.colors[track_idx % len(cmap.colors)]

        mlab_points.append(mlab.points3d(points[:, 0], points[:, 1], points[:, 2], color=color[:3], scale_factor=scale_factor))
        mlab_lines.append(draw_all_lines(points, scheme, bodypart_dict))

    return fig, mlab_points, mlab_lines


def get_frame_3d(fig: Any, all_points: np.ndarray, mlab_points: List[Any], mlab_lines: List[Any], 
                 scheme: List[List[str]], bodypart_dict: Dict[str, int], show_lines: bool = True) -> np.ndarray:
    """Get frame for 3D visualization.
    
    Args:
        fig: Mayavi figure.
        all_points: Array containing all 3D points with shape of (n_tracks, n_bodyparts, [x, y, z] coordinates)
        mlab_points: List of point objects.
        mlab_lines: List of line objects.
        scheme: List of lists where each inner list contains body parts to be connected.
        bodypart_dict: Dictionary mapping body part names to indices.
        show_lines: Whether to draw lines between points. If not, it can save lots of time.

    Returns:
        Screenshot of the 3D visualization.
    """
    fig.scene.disable_render = True
    for points, mlab_point, mlab_line in zip(all_points, mlab_points, mlab_lines):
        mlab_point.mlab_source.points = points
        if show_lines: update_all_lines(mlab_line, points, scheme, bodypart_dict)
    fig.scene.disable_render = False
    # mlab.show()
    return mlab.screenshot(antialiased=True)


def write_video(config: Dict[str, Any], labels_filepaths: List[str], output_filepath: str, 
                room_dimensions: List[int], fps: int = 25, colormap_name: str = 'Set2', 
                verbose: bool = True) -> None:
    """Write 3D visualization to video.

    Args:
        config: Configuration dictionary.
        labels_filepaths: List of filenames containing label information.
        output_filepath: Output filename for the video.
        room_dimensions: Room dimensions [width, length, height, grid_size].
        fps: Frames per second for the video. Defaults to 25.
        colormap_name: Name of the colormap. Defaults to 'Set2'.
        verbose: Whether to print progress. Defaults to True.
    """
    bodyparts = config['bodyparts']
    bodypart_dict = dict(zip(bodyparts, range(len(bodyparts))))
    scheme = config['visualization']['scheme']

    all_points, all_scores, all_errors, rotation_matrix, axes_3d = load_pose_3d(labels_filepaths, bodyparts)
    n_tracks, n_bodyparts, n_frames, _ = all_points.shape

    fig, mlab_points, mlab_lines = initialize_3d(n_tracks, n_bodyparts, scheme, bodypart_dict, axes_3d, 1, room_dimensions, colormap_name)
    
    writer = skvideo.io.FFmpegWriter(output_filepath, inputdict={'-framerate': str(fps)},
                                     outputdict={'-vcodec': 'libx264', '-pix_fmt': 'yuv420p', '-preset': 'superfast', '-crf': '23'})

    iterator = trange(n_frames, ncols=100, desc='3D Visualizing... ', unit='frames') if verbose else range(n_frames)

    for frame_idx in iterator:
        img = get_frame_3d(fig, all_points[:, :, frame_idx], mlab_points, mlab_lines, scheme, bodypart_dict, show_lines=True)
        writer.writeFrame(img)

    writer.close()
    mlab.close(fig)


def generate_video_3d(config: Dict[str, Any], room_dimensions: List[int] = [680, 980, 780, 30], 
                      filtered: bool = False, verbose: bool = True) -> None:
    """Generate video with triangulated 3d poses.

    Args:
        config: Configuration dictionary.
        room_dimensions: Room dimensions [width, length, height, grid_size]. Defaults to [680, 980, 780, 30].
        filtered: If True, filtered poses are used. Defaults to False.
        verbose: Whether to print progress. Defaults to True.
    """
    project_dir = config['project_dir']
    videos_raw_dir = config['directory']['videos_raw']
    video_extension = config['video_extension']
    
    poses_3d_dir = config['directory']['poses_3d_filtered'] if filtered else config['directory']['poses_3d']
    videos_labeled_3d_dir = config['directory']['videos_labeled_3d_filtered'] if filtered else config['directory']['videos_labeled_3d']

    output_dir = os.path.join(project_dir, videos_labeled_3d_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, '3d.mp4')

    video_filepaths = glob(os.path.join(project_dir, videos_raw_dir, f"*.{video_extension}"))
    video_params = get_video_params(video_filepaths[0])

    labels_filepaths = sorted(glob(os.path.join(project_dir, poses_3d_dir, '*.csv')))

    write_video(config, labels_filepaths, output_filepath, room_dimensions, video_params['fps'], verbose=verbose)
