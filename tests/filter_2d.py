import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

from marmopose.utils.common import load_pose_2d, save_pose_2d


def my_filter(config: dict, all_points: np.ndarray):
    """Simplest filter.
    
    TODO: Apply marmoset body model
    """
    n_tracks, n_frames, n_bodyparts, _ = all_points.shape
    points = np.full((n_tracks, n_frames, n_bodyparts, 2), np.nan, dtype='float64')
    scores = np.empty((n_tracks, n_frames, n_bodyparts), dtype='float64')

    for track_idx, track_points in enumerate(all_points):
        points_full = track_points[:, :, :2].squeeze()
        scores_full = track_points[:, :, 2].squeeze()

        # Filter
        bad = scores_full < config['filter']['score_threshold']
        points_full[bad] = np.nan
        # scores_full[bad] = 2.0 # 2.0 mark low-confidence point scores
        
        # Fill in missing value
        for frame in range(n_frames):
            idx = np.argwhere(np.isnan(points_full[frame, :, 0]))
            if frame == 0:
                for i in idx:
                    for j in range(1, n_frames):
                        if not np.isnan(points_full[j][i]).any():
                            points_full[frame][i] = points_full[j][i]
                            break
            else:
                if len(idx) == scores_full.shape[1]:
                    points_full[frame] = points_full[frame-1]
                else:
                    move = np.nanmean(points_full[frame] - points_full[frame-1], axis=0, keepdims=True)
                    points_full[frame][idx] = points_full[frame-1][idx] + move if np.isnan(move).all() else points_full[frame-1][idx]

        points[track_idx] = points_full
        scores[track_idx] = scores_full
    
    all_points = np.concatenate((points, scores[:,:,:,np.newaxis]), axis=3)
    return all_points


FILTER_MAPPING = {
    'myfilter': my_filter
}
POSSIBLE_FILTERS = FILTER_MAPPING.keys()


def filter_2d(config):
    """Filter and fill in missing data of the predicted 2d poses."""
    project_path = config['project_path']
    poses_2d_folder = config['folder']['poses_2d']
    poses_2d_filtered_folder = config['folder']['poses_2d_filtered']
    filter_types = config['filter']['type']
    if not isinstance(filter_types, list):
        filter_types = [filter_types]

    for filter_type in filter_types:
        assert filter_type in POSSIBLE_FILTERS, \
            "Invalid filter type, should be one of {}, but found {}".format(POSSIBLE_FILTERS, filter_type)

    labels_list = sorted(glob(os.path.join(project_path, poses_2d_folder, '*.h5')))
    output_folder = os.path.join(project_path, poses_2d_filtered_folder)
    os.makedirs(output_folder, exist_ok=True)

    for labels_fname in labels_list:
        basename = os.path.basename(labels_fname)
        output_fname = os.path.join(project_path, poses_2d_filtered_folder, basename)

        if os.path.exists(output_fname):
            print(f'{output_fname} already exists!')
            continue
        
        all_points, metadata = load_pose_2d(labels_fname)

        for filter_type in filter_types:
            filter_fun = FILTER_MAPPING[filter_type]
            all_points = filter_fun(config, all_points)

        save_pose_2d(all_points, metadata, output_fname)
    print(f'Filtered 2D poses stored in: {output_folder}')


def plot_filter(config, bodyparts_idx=range(8), mode='seperate'):
    """Plot filtered results"""
    project_dir = config['project_dir']
    poses_2d_dir = config['directory']['poses_2d']
    poses_2d_filtered_dir = config['directory']['poses_2d_filtered']

    labels_list = sorted(glob(os.path.join(project_dir, poses_2d_dir, '*.h5')))
    
    for labels_fname in labels_list:
        basename = os.path.basename(labels_fname)
        print(f'Plot file: {basename}')
        points, metadata = load_pose_2d(labels_fname)
        filtered_fname = os.path.join(project_dir, poses_2d_filtered_dir, basename)
        points_filtered, metadata = load_pose_2d(filtered_fname)

        n_tracks = points.shape[0]
        bodyparts = metadata['bodyparts']
        tracks = metadata['tracks']
        
        if mode == 'seperate':
            plt.figure(figsize=(12, 16))
            for track_idx in range(n_tracks):
                plt.subplot(n_tracks*2, 2, 1+track_idx*4)
                plt.title(f'{tracks[track_idx]} original pose x')
                for bp_idx in bodyparts_idx:
                    plt.plot(points[track_idx, :, bp_idx, 0], label=f'{bodyparts[bp_idx]}')
                plt.ylim([0, 1900])
                plt.legend(prop={'size': 8})

                plt.subplot(n_tracks*2, 2, 2+track_idx*4)
                plt.title(f'{tracks[track_idx]} original pose y')
                for bp_idx in bodyparts_idx:
                    plt.plot(points[track_idx, :, bp_idx, 1], label=f'{bodyparts[bp_idx]}')
                plt.ylim([0, 1900])
                plt.legend(prop={'size': 8})

                plt.subplot(n_tracks*2, 2, 3+track_idx*4)
                plt.title(f'{tracks[track_idx]} filtered pose x')
                for bp_idx in bodyparts_idx:
                    plt.plot(points_filtered[track_idx, :, bp_idx, 0], label=f'{bodyparts[bp_idx]}')
                plt.ylim([0, 1900])
                plt.legend(prop={'size': 8})
                
                plt.subplot(n_tracks*2, 2, 4+track_idx*4)
                plt.title(f'{tracks[track_idx]} filtered pose y')
                for bp_idx in bodyparts_idx:
                    plt.plot(points_filtered[track_idx, :, bp_idx, 1], label=f'{bodyparts[bp_idx]}')
                plt.ylim([0, 1900])
                plt.legend(prop={'size': 8})
            plt.show()
        elif mode == 'x-y':
            plt.figure(figsize=(12,8))
            for track_idx in range(n_tracks):
                plt.subplot(n_tracks, 2, 1+track_idx*2)
                plt.title(f'{tracks[track_idx]} original pose')
                for bp_idx in bodyparts_idx:
                    plt.plot(points[track_idx, :, bp_idx, 0], points[track_idx, :, bp_idx, 1], label=f'{bodyparts[bp_idx]}')
                plt.legend(prop={'size': 8})

                plt.subplot(n_tracks, 2, 2+track_idx*2)
                plt.title(f'{tracks[track_idx]} filtered pose')
                for bp_idx in bodyparts_idx:
                    plt.plot(points_filtered[track_idx, :, bp_idx, 0], points_filtered[track_idx, :, bp_idx, 1], label=f'{bodyparts[bp_idx]}')
                plt.legend(prop={'size': 8})
            plt.show()