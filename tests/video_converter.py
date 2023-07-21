import os
import numpy as np
import tempfile
from multiprocessing import Pool
from glob import glob


# def convert_videos(video_dir):
#     """
#     Convert and merge videos at the same time.
#     """
#     video_files = sorted(glob(os.path.join(video_dir, '*.mp4')))
#     output_file = video_dir.split(' ')[0] + ".mp4"

#     with tempfile.NamedTemporaryFile(mode='w', delete=True) as tmp:
#         print(tmp.name)
#         for file in video_files:
#             tmp.write(f"file '{file}'\n")
#         tmp.seek(0)

#         command = f'ffmpeg -f concat -safe 0 -i {tmp.name} -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 23 {output_file}'
#         os.system(command)


def convert_videos(video_dir):
    """
    Convert each part of videos first and then merge them.
    """
    video_files = sorted(glob(os.path.join(video_dir, '*.mp4')))
    output_file = video_dir.split('_')[0] + "-raw.mp4"
    
    converted_files = []
    for file in video_files:
        converted_file = os.path.splitext(file)[0] + '_converted.mp4'
        command = f'ffmpeg -y -i {file} -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 23 {converted_file}'
        os.system(command)
        converted_files.append(converted_file)

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
        print(tmp.name)
        for file in converted_files:
            tmp.write(f"file '{file}'\n")
        tmp.close()

    command = f'ffmpeg -f concat -safe 0 -i {tmp.name} -c copy {output_file}'
    os.system(command)


def rename_files_and_directories(root_dir, source='local'):
    """
    Rename all files and directories, here replace all ' ' with '_'.
    """
    for path, dirs, files in os.walk(root_dir, topdown=False):
        for filename in files:
            old_name = os.path.join(path, filename)
            new_name = os.path.join(path, filename.replace(' ', '_'))
            os.rename(old_name, new_name)

        for dirname in dirs:
            old_name = os.path.join(path, dirname)
            new_name = os.path.join(path, dirname.replace(' ', '_'))
            os.rename(old_name, new_name)


def trim_video(input_file, hour, minute, second, frame, duration):
    """
    Trim target video at start time with duration time.
    """
    start = time2s(hour, minute, second, frame)
    output_file = os.path.splitext(input_file)[0] + f'_{start}-{duration}.mp4'
    # command = f'ffmpeg -i {input_file} -vf "select=between(n\,{start_time}\,{duration_time})" -vsync 0 {output_file}'
    command = f'ffmpeg -ss {start} -t {duration} -i {input_file} -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 23 {output_file}'
    os.system(command)


def time2s(hour, minute, second, frame, fps=25):
    """
    Convert target time and frame number to time in seconds.
    """
    return (((hour*60+minute)*60+second)*25+frame-1)/fps


if __name__ == '__main__':
    # rename_files_and_directories('D:/Chengchaoqun/20230718')ß

    # video_dir = '/Users/leosword/Documents/Research/Videos/20230511/Single/bak-3-2/'
    # with Pool() as p:
    #     cameras = [video_dir+'bak-1-2_1.240_20230615163726', 
    #                video_dir+'bak-2-2_1.242_20230615163726', 
    #                video_dir+'bak-3-2_1.244_20230615163726',
    #                video_dir+'bak-4-2_1.246_20230615163726',
    #                video_dir+'bak-5-2_1.248_20230615163725',
    #                video_dir+'bak-6-2_1.250_20230615163725']
    #     p.map(convert_videos, cameras)

    video_dir = 'D:/Chengchaoqun/20230718/'
    with Pool() as p:
        params = [(video_dir+'bak-1-2-raw.mp4', 0, 10, 7, 6, 4500),
                  (video_dir+'bak-2-2-raw.mp4', 0, 10, 5, 8, 4500),
                  (video_dir+'bak-3-2-raw.mp4', 0, 10, 5, 15, 4500),
                  (video_dir+'bak-4-2-raw.mp4', 0, 10, 8, 13, 4500),
                  (video_dir+'bak-5-2-raw.mp4', 0, 1, 42, 0, 4500),
                  (video_dir+'bak-6-2-raw.mp4', 0, 1, 42, 0, 4500)]
        p.starmap(trim_video, params)



