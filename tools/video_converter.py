import os
import re
import tempfile
from pathlib import Path
from multiprocessing import Pool
from typing import List


def rename_files_and_directories(root_dir: Path) -> None:
    """
    Rename all files and directories within the specified root directory, replacing all spaces with underscores.
    
    Args:
        root_dir: The root directory path as a pathlib. Path object.
    """
    for path in root_dir.rglob('*'):
        if path.stem.startswith('._'):
            path.unlink()
            break
        if ' ' in path.name:
            new_name = path.name.replace(' ', '_')
            path.rename(path.with_name(new_name))


def convert_video(file_path: Path, output_path: Path) -> None:
    """
    Convert a video file to a specific format using FFmpeg.
    
    Args:
        file_path: The path to the input video file.
        output_path: The path where the converted video file should be saved.
    """
    # command = f'ffmpeg -y -i {file_path} -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 23 {output_path}'
    command = f'ffmpeg -y -i {file_path} -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 23 -vf "scale=1920:1080, mpdecimate, setpts=N/FRAME_RATE/TB" {output_path}'
    os.system(command)


def merge_videos(files_list: List[Path], output_file: Path) -> None:
    """
    Merge multiple video files into a single video file using FFmpeg.
    
    Args:
        files_list: A list of paths to the video files to be merged.
        output_file: The path where the merged video file should be saved.
    """
    print(files_list)
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
        for file in files_list:
            tmp.write(f"file '{file}'\n")
        tmp.close()

    command = f'ffmpeg -f concat -safe 0 -i {tmp.name} -c copy {output_file}'
    os.system(command)
    os.remove(tmp.name)


def convert_videos(video_dir: Path, do_marge: bool=True) -> None:
    """
    Convert each part of videos first and then merge them.
    
    Args:
        video_dir: The directory containing the video files to be converted and merged.
        do_marge: Whether to merge all of the converted videos in the directory.
    """
    video_files = sort_video_files(video_dir.glob('*.mp4'))
    video_files = [file for file in video_files if 'converted' not in file.name]
    
    converted_files = []
    for file in video_files:
        print(file)
        converted_file = file.parent / (file.stem + '_converted.mp4')
        if not converted_file.exists():
            convert_video(file, converted_file)
        else:
            print(f'Already converted: {converted_file}')
        converted_files.append(converted_file)
    
    output_file = video_dir.parent / 'raw' / (video_dir.name.split('_')[0] + "-raw.mp4")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if do_marge:
        merge_videos(converted_files, output_file)


def get_video_file_number(file_name: str) -> int:
    """
    Extract the video file number from the file name.

    Args:
        file_name: The file name.

    Returns:
        The video file number.
    """
    match = re.search(r'_(\d+)(?:_converted)?\.mp4$', file_name)
    if match:
        return int(match.group(1))
    return 0


def sort_video_files(video_files: List[Path]) -> List[Path]:
    """
    Sort a list of video files based on the video file number.

    Args:
        video_files: The list of video files to sort.

    Returns:
        The sorted list of video files.
    """
    return sorted(video_files, key=lambda x: get_video_file_number(x.name))


def get_camera_dirs(base_dir: Path) -> List[Path]:
    """
    Get a list of camera directories from the base directory.
    
    Args:
        base_dir: The base directory containing camera directories.
    
    Returns:
        A sorted list of paths to camera directories.
    """
    return sorted([cam_dir for cam_dir in base_dir.iterdir() if cam_dir.is_dir()])


def trim_video(input_file, hour, minute, second, frame, duration):
    """
    Trim target video at start time with duration time.
    """
    start = time2s(hour, minute, second, frame)
    # output_file = os.path.splitext(input_file)[0].strip('-raw') + '.mp4'
    output_file = input_file.parent.parent / (input_file.stem.strip('-raw') + '.mp4')
    # print(output_file)
    # command = f'ffmpeg -i {input_file} -vf "select=between(n\,{start_time}\,{duration_time})" -vsync 0 {output_file}'
    command = f'ffmpeg -ss {start} -t {duration} -i {input_file} -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 23 {output_file}'
    os.system(command)


def time2s(hour, minute, second, frame, fps=25):
    """
    Convert target time and frame number to time in seconds.
    """
    return (((hour*60+minute)*60+second)*25+frame-1)/fps


if __name__ == '__main__':
    """
    Use multiprocessing to convert and combine raw videos.
    TODO: Specify the directory containing the raw videos.
    """
    videos_dir = Path(r'D:\ccq\Videos\20240411_test')
    rename_files_and_directories(videos_dir)

    do_marge = True # NOTE: Whether to merge all of the converted videos in each sub-directory. You might need to set it to False if the videos are too long.
    args = [(cam_dir, do_marge) for cam_dir in get_camera_dirs(videos_dir)]

    with Pool() as p:
        p.starmap(convert_videos, args)


    """
    Use this block to align videos. It uses multiprocessing to trim videos at specific time points.
    TODO: Specify the start time and duration for each video.
    """
    # duration = 3 #11:15:10
    # with Pool() as p:
    #     params = [(videos_dir / 'raw/323-1-2-raw.mp4', 0, 8, 2, 1, duration),
    #               (videos_dir / 'raw/323-2-2-raw.mp4', 0, 8, 3, 5, duration),
    #               (videos_dir / 'raw/323-3-2-raw.mp4', 0, 19, 31, 7, duration),
    #               (videos_dir / 'raw/323-4-2-raw.mp4', 0, 19, 32, 1, duration),
    #               (videos_dir / 'raw/bak-5-2-raw.mp4', 0, 12, 16, 4, duration),
    #               (videos_dir / 'raw/bak-6-2-raw.mp4', 0, 12, 16, 23, duration)]
    #     p.starmap(trim_video, params)