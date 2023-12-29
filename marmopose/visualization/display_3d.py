import skvideo.io
import numpy as np
from tqdm import trange
from pathlib import Path
from typing import Dict, Any, List, Tuple
from mayavi import mlab
mlab.options.offscreen = True

from marmopose.utils.data_io import load_points_3d_h5
from marmopose.utils.helpers import get_color_list


def generate_video_3d(config: Dict[str, Any], points_3d_source: str = 'original', fps: int = 25, verbose: bool = True) -> None:
    """
    Generates a 3D video visualization based on 3D points.
    
    Args:
        config: Configuration dictionary containing directories, body parts, visualization settings, etc.
        points_3d_source (optional): The source of 3D points. Either 'original' or 'optimized'. Defaults to 'original'.
        fps (optional): Frames per second for the output video. Defaults to 25.
        verbose (optional): Whether to print verbose output. Defaults to True.
        
    Raises:
        AssertionError: If the points_3d_source is invalid.
    """
    assert points_3d_source in ['original', 'optimized'], f'Invalid points_3d_source, must be one of: original, optimized'

    project_dir = Path(config['directory']['project'])
    points_3d_path = project_dir / config['directory']['points_3d'] / f'{points_3d_source}.h5'
    videos_labeled_3d_path = project_dir / config['directory']['videos_labeled_3d'] / f'{points_3d_source}.mp4'
    videos_labeled_3d_path.parent.mkdir(parents=True, exist_ok=True)

    bodyparts = config['animal']['bodyparts']
    skeleton = config['visualization']['skeleton']
    skeleton_indices = [[bodyparts.index(bp) for bp in line] for line in skeleton]

    track_color_list = get_color_list(config['visualization']['track_cmap'], config['animal']['number'], cvtInt=False)
    skeleton_color_list = get_color_list(config['visualization']['skeleton_cmap'], len(skeleton_indices), cvtInt=False)

    all_points_3d = load_points_3d_h5(points_3d_path, verbose=verbose) # (n_tracks, n_frames, n_bodyparts, (x, y, z))
    render_video_with_pose_3d(all_points_3d, videos_labeled_3d_path, skeleton_indices, track_color_list, skeleton_color_list, fps, verbose)


def render_video_with_pose_3d(all_points_3d: np.ndarray, output_filepath: Path, 
                              skeleton_indices: List[List[int]], 
                              track_color_list: List[Tuple[float, float, float]], 
                              skeleton_color_list: List[Tuple[float, float, float]], 
                              fps: int = 25, verbose: bool = True) -> None:
    """
    Renders video with 3D poses.
    
    Args:
        all_points_3d: An array containing the 3D coordinates. Shape of (n_tracks, n_frames, n_bodyparts, 3), final channel (x, y, z).
        output_filepath: File path for saving the output video.
        skeleton_indices: Indices of the skeleton joints.
        track_color_list: List of RGB colors for tracks.
        skeleton_color_list: List of RGB colors for skeletons.
        fps (optional): Frames per second for the output video. Defaults to 25.
        verbose (optional): Whether to print verbose output. Defaults to True.
    """
    n_tracks, n_frames, n_bodyparts, _ = all_points_3d.shape

    fig, mlab_points, mlab_lines = initialize_3d(n_tracks, n_bodyparts, skeleton_indices, track_color_list, skeleton_color_list)

    writer = skvideo.io.FFmpegWriter(output_filepath, inputdict={'-framerate': str(fps)},
                                    outputdict={'-vcodec': 'libx264', '-pix_fmt': 'yuv420p', '-preset': 'superfast', '-crf': '23'})

    iterator = trange(n_frames, ncols=100, desc='3D Visualizing... ', unit='frames') if verbose else range(n_frames)

    for frame_idx in iterator:
        img = get_image_3d(fig, all_points_3d[:, frame_idx], mlab_points, mlab_lines, skeleton_indices, show_lines=True)
        writer.writeFrame(img)

    writer.close()
    mlab.close(fig)


def initialize_3d(n_tracks: int, n_bodyparts: int, skeleton_indices: List[List[int]], 
                  track_color_list: List[Tuple[float, float, float]], 
                  skeleton_color_list: List[Tuple[float, float, float]], 
                  axes: np.ndarray = None, scale: float = 1.0, 
                  room_dimensions: List[int] = [730, 1030, 860, 30]) -> Tuple[mlab.figure, List[mlab.points3d], List[mlab.plot3d]]:
    """
    Initialize the 3D figure and return it along with the points and lines.
    
    Args:
        n_tracks: Number of tracks.
        n_bodyparts: Number of body parts.
        skeleton_indices: Indices of the skeleton joints.
        track_color_list: List of RGB colors for tracks.
        skeleton_color_list: List of RGB colors for skeletons.
        axes (optional): Axes for the plot. Defaults to None.
        scale (optional): Scale factor for the plot dimensions. Defaults to 1.0.
        room_dimensions (optional): Dimensions of the room for the grid. Defaults to [680, 980, 780, 30].
        
    Returns:
        A tuple containing the figure, points, and lines.
    """
    fig = mlab.figure(bgcolor=(1, 1, 1), size=(int(1920*scale), int(1080*scale)))
    fig.scene.anti_aliasing_frames = 8
    mlab.clf()

    # TODO: How to define room dimensions?****************************************************************
    # Draw room with grids
    # mlab.view(135, 120)
    draw_room_grids(*room_dimensions)

    # Draw axes
    scale_factor = np.max(room_dimensions)//50
    if axes is not None:
        mlab.points3d(axes[:, 0], axes[:, 1], axes[:, 2], color=(0.8, 0.8, 0.8), scale_factor=scale_factor)
    #****************************************************************************************************

    # Initialize points and lines
    mlab_points, mlab_lines = [], []
    for track_idx in range(n_tracks):
        points = np.zeros((n_bodyparts, 3))

        mlab_points.append(mlab.points3d(points[:, 0], points[:, 1], points[:, 2], color=track_color_list[track_idx], scale_factor=scale_factor))
        mlab_lines.append(draw_lines(points, skeleton_indices, skeleton_color_list))

    return fig, mlab_points, mlab_lines


def get_image_3d(fig: mlab.figure, 
                 all_points: np.ndarray, 
                 mlab_points: List[mlab.points3d], 
                 mlab_lines: List[mlab.plot3d],
                 skeleton_indices: List[List[int]], 
                 show_lines: bool = True) -> np.ndarray:
    """
    Capture a 3D image frame with updated pose points and lines.
    
    Args:
        fig: Mayavi figure object for rendering.
        all_points: Array of shape (n_tracks, n_bodyparts, 3) containing points in 3D.
        mlab_points: List of Mayavi point objects for each track.
        mlab_lines: List of Mayavi line objects for each track.
        skeleton_indices: Indices of body parts that are connected by lines in the skeleton.
        show_lines (optional): Whether to show skeleton lines. Defaults to True.
    
    Returns:
        Captured image frame.
    """
    fig.scene.disable_render = True
    for points, mlab_point, mlab_line in zip(all_points, mlab_points, mlab_lines):
        mlab_point.mlab_source.points = points
        if show_lines: update_lines(mlab_line, points, skeleton_indices)
    fig.scene.disable_render = False
    # mlab.show()
    return mlab.screenshot(antialiased=True)


def update_lines(lines: List[Any], points: np.ndarray, skeleton_indices) -> None:
    for line, bodyparts_indices in zip(lines, skeleton_indices):
        line.mlab_source.points = points[bodyparts_indices]


def draw_room_grids(width: int, length: int, height: int, grid_size: int) -> None:
    """
    Draw grids on the room floor and walls.
    
    Args:
        width: Width of the room.
        length: Length of the room.
        height: Height of the room.
        grid_size: Size of the grid squares.
    """
    col = (1, 1, 1)
    x, y = np.mgrid[:width:grid_size, :length:grid_size]
    mlab.surf(x, y, np.zeros(x.shape), representation='wireframe', color=col, line_width=0.5)
    z, y = np.mgrid[:height:grid_size, :length:grid_size]
    mlab.surf(np.zeros(y.shape), y, z, representation='wireframe', color=col, line_width=0.5)
    x, z = np.mgrid[:width:grid_size, :height:grid_size]
    mlab.surf(x, np.zeros(z.shape), z, representation='wireframe', color=col, line_width=0.5)


def draw_lines(points: np.ndarray, 
               skeleton_indices: List[List[int]], 
               skeleton_color_list: List[Tuple[float, float, float]]) -> List[mlab.plot3d]:
    """
    Draw all lines connecting body parts based on the skeleton configuration.
    
    Args:
        points: Array of 3D coordinates of body parts.
        skeleton_indices: List of index pairs indicating which body parts to connect.
        skeleton_color_list: List of RGB colors for each skeleton line.
    
    Returns:
        List of created line objects.
    """
    lines = []
    for idx, bodyparts_indices in enumerate(skeleton_indices):
        line = mlab.plot3d(points[bodyparts_indices, 0], points[bodyparts_indices, 1], points[bodyparts_indices, 2],
                           np.ones(len(bodyparts_indices)), reset_zoom=False,
                           color=skeleton_color_list[idx], tube_radius=None, line_width=5)
        lines.append(line)

    return lines