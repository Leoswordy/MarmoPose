import logging
from pathlib import Path

import cv2
import torch
import numpy as np
from tqdm import trange

from marmopose.config import Config
from marmopose.calibration.cameras import CameraGroup
from marmopose.processing.autoencoder import DaeTrainer
from marmopose.processing.optimization import optimize_coordinates
from marmopose.utils.data_io import load_points_bboxes_2d_h5, save_points_3d_h5
from marmopose.utils.helpers import Timer

logger = logging.getLogger(__name__)


class Reconstructor3D:
    def __init__(self, config: Config):
        self.init_dir(config)

        self.camera_group = CameraGroup.load_from_json(self.cam_params_path)
        self.pair_F_matrices = compute_all_fundamental_matrices(self.camera_group)
        logger.info(f'Loaded camera group from: {self.cam_params_path}')

        self.n_tracks = config.animal['n_tracks']
        self.skip_index = [config.animal['bodyparts'].index(bp) for bp in config.animal['skip']]
        self.config = config

        if self.config.triangulation['dae_enable']:
            self.build_dae_model(config.directory['dae_model'])
            logger.info(f'Enabled filling of missing values using Denoising Autoencoder')

    def init_dir(self, config):
        self.cam_params_path = Path(config.sub_directory['calibration']) / 'camera_params.json'
        self.points_2d_path = Path(config.sub_directory['points_2d']) / 'original.h5'
        self.points_3d_path = Path(config.sub_directory['points_3d']) / 'original.h5'
        self.points_3d_optimized_path = Path(config.sub_directory['points_3d']) / 'optimized.h5'
    
    def build_dae_model(self, dae_model_dir):
        checkpoint_files = list(Path(dae_model_dir).glob('*best*.pth'))
        assert len(checkpoint_files) == 1, f'Zero/Multiple best checkpoint files found in {dae_model_dir}'

        dae_checkpoint = str(checkpoint_files[0])
        logger.info(f'Loaded DAE from: {dae_checkpoint}')
        
        dae = torch.load(dae_checkpoint)
        self.dae_trainer = DaeTrainer(model=dae, bodyparts=self.config.animal['bodyparts'])

    def triangulate(self, all_points_with_score_2d: np.ndarray = None, all_bboxes: np.ndarray = None, reassign_id: bool = False):
        """Triangulate 2D points to 3D coordinates, with optional ID reassignment.

        Args:
            all_points_with_score_2d (np.ndarray): 2D points with scores.
            all_bboxes (np.ndarray): Bounding boxes.
            reassign_id (bool): Flag to enable ID reassignment.
        """
        if all_points_with_score_2d is None:
            all_points_with_score_2d, all_bboxes = load_points_bboxes_2d_h5(self.points_2d_path)

        if reassign_id:
            logger.info("ID reassignment enabled")
            all_points_3d, all_points_with_score_2d = self._triangulate_with_reassignment(all_points_with_score_2d) # TODO: Store reassigned all_points_with_score_2d and all_bboxes
        else:
            all_points_3d = self._triangulate_without_reassignment(all_points_with_score_2d)

        all_points_3d = filter_outliers_by_skeleton(all_points_3d, 150) # filter out outliers

        for track_idx in range(self.n_tracks):
            track_name = f'track{track_idx+1}'
            points_3d = all_points_3d[track_idx]  # (n_frames, n_bodyparts, 3)

            save_points_3d_h5(points = points_3d, name = track_name, file_path = self.points_3d_path)

            if self.config.triangulation['dae_enable']:
                logger.info(f'Optimize {track_name}')
                points_3d = self.fill_with_dae(points_3d)
            
            if self.config.optimization['do_optimize']:
                points_3d_optimized = optimize_coordinates(self.config, self.camera_group, points_3d, all_points_with_score_2d[:, track_idx])
                points_3d_optimized[:, self.skip_index] = np.nan
                save_points_3d_h5(points = points_3d_optimized, name = track_name, file_path = self.points_3d_optimized_path)
                
    def _triangulate_without_reassignment(self, all_points_with_score_2d):
        n_cams, n_tracks, n_frames, n_bodyparts, n_dim = all_points_with_score_2d.shape

        all_points_3d = np.full((n_tracks, n_frames, n_bodyparts, 3), np.nan)

        for frame_idx in trange(n_frames, ncols=100, desc='Triangulating... ', unit='frames'):
            all_points_with_score_2d_frame = all_points_with_score_2d[:, :, frame_idx]  # (n_cams, n_tracks, n_bodyparts, 3)
            points_3d = self.triangulate_frame(all_points_with_score_2d_frame, ransac=True)  # (n_tracks, n_bodyparts, 3)
            all_points_3d[:, frame_idx] = points_3d
        
        return all_points_3d

    def _triangulate_with_reassignment(self, all_points_with_score_2d):
        n_cams, n_tracks, n_frames, n_bodyparts, n_dim = all_points_with_score_2d.shape

        all_points_3d_reassigned = np.full((n_tracks, n_frames, n_bodyparts, 3), np.nan)
        all_points_with_score_2d_reassigned = all_points_with_score_2d.copy()

        for frame_idx in trange(n_frames, ncols=100, desc='Triangulating... ', unit='frames'):
            all_points_with_score_2d_frame = all_points_with_score_2d[:, :, frame_idx]  # (n_cams, n_tracks, n_bodyparts, 3)

            all_points_with_score_2d_frame_undistorted = get_undistorted_points(self.camera_group, all_points_with_score_2d_frame) # Normalized
            final_ids = process_frame(self.pair_F_matrices, self.camera_group, all_points_with_score_2d_frame_undistorted)
            points_with_score_2d = prepare_points_with_score_2d(all_points_with_score_2d_frame_undistorted, final_ids, n_bodyparts, n_cams)

            points_with_score_2d_distorted = points_with_score_2d.copy() # Distorted 
            
            for i, (points_with_score_2d_cam, cam) in enumerate(zip(points_with_score_2d, self.camera_group.cameras)):
                undistorted = points_with_score_2d_cam.reshape(-1, 3)[:, :2]
                distorted = cam.distort_points(undistorted).reshape(n_tracks, n_bodyparts, 2)
                points_with_score_2d_distorted[i, :, :, :2] = distorted
            
            points_3d = self.triangulate_frame(points_with_score_2d_distorted, ransac=True)  # (n_tracks, n_bodyparts, 3)

            all_points_3d_reassigned[:, frame_idx] = points_3d
            all_points_with_score_2d_reassigned[:, :, frame_idx] = points_with_score_2d_distorted
        
        return all_points_3d_reassigned, all_points_with_score_2d_reassigned

    def triangulate_frame(self, points_with_score_2d: np.ndarray, ransac=True):
        """
        Args:
            points_with_score_2d: (n_cams, n_tracks, n_bodyparts, (x, y, score))
        
        Returns:
            points_3d: (n_tracks, n_bodyparts, (x, y, z))
        """
        n_cams, n_tracks, n_bodyparts, n_dim = points_with_score_2d.shape

        points_with_score_2d_flat = points_with_score_2d.reshape(n_cams, n_tracks*n_bodyparts, n_dim)

        if ransac:
            points_3d_flat = self.camera_group.triangulate_ransac(points_with_score_2d_flat, undistort=True)
        else:
            points_3d_flat = self.camera_group.triangulate(points_with_score_2d_flat, undistort=True)
            
        points_3d = points_3d_flat.reshape((n_tracks, n_bodyparts, 3)) # (n_tracks, n_bodyparts, (x, y, z))

        return points_3d

    def fill_with_dae(self, points_3d: np.ndarray, batch_size: int = 7500) -> np.ndarray:
        """Fills missing values in 3D coordinates using Denoising Autoencoder.

        Args:
            points_3d (np.ndarray): Input 3D points with shape (n_frames, n_bodyparts, 3).
            batch_size (int): Number of frames to process in each batch.

        Returns:
            np.ndarray: The filled 3D points with missing values imputed.
        """
        n_frames = points_3d.shape[0]
        filled_points_3d = np.empty_like(points_3d)

        num_batches = int(np.ceil(n_frames / batch_size))

        with trange(n_frames, ncols=100, desc="Filling with DAE...", unit="frames") as progress_bar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_frames)

                batch_points = points_3d[start_idx:end_idx]
                batch_filled_points = batch_points.copy()

                mask_invalid = np.isnan(batch_points)
                res = self.dae_trainer.predict(batch_points)
                batch_filled_points[mask_invalid] = res[mask_invalid]
                filled_points_3d[start_idx:end_idx] = batch_filled_points

                progress_bar.update(end_idx - start_idx)

        return filled_points_3d  # (n_frames, n_bodyparts, 3)


def filter_outliers_by_skeleton(data: np.ndarray, threshold: float = 150) -> np.ndarray:
    """
    Filters out outliers in a 3D coordinate array based on the skeleton structure.

    Args:
        data (np.ndarray): Input array of shape (n_tracks, n_frames, n_bodyparts, 3).
        max_length (float): The maximum plausible distance between keypoints.

    Returns:
        np.ndarray: The filtered 3D coordinates array with outliers set to NaN.
    """
    filtered_data = np.copy(data)
    
    n_tracks, n_frames, n_bodyparts, _ = data.shape

    for track in range(n_tracks):
        for frame in range(n_frames):
            keypoints = data[track, frame]
            distances = np.linalg.norm(keypoints[:, np.newaxis] - keypoints[np.newaxis, :], axis=2)

            for bodypart in range(n_bodyparts):
                if bodypart == n_bodyparts - 2: # tailmid
                    outlier_count = np.sum(distances[bodypart] > 2*threshold)
                elif bodypart == n_bodyparts - 1: # tailend
                    outlier_count = np.sum(distances[bodypart] > 3*threshold)
                else: # other bodyparts
                    outlier_count = np.sum(distances[bodypart] > threshold)
                
                if outlier_count > (n_bodyparts // 2):
                    filtered_data[track, frame, bodypart] = np.nan

    return filtered_data


def skew_symmetric(v):
    """Returns the skew-symmetric matrix of vector v."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def compute_fundamental_matrix(cam1, cam2):
    """Computes the fundamental matrix between two cameras."""
    K1 = cam1.matrix
    K2 = cam2.matrix

    R1, _ = cv2.Rodrigues(cam1.rotation)
    R2, _ = cv2.Rodrigues(cam2.rotation)

    t1 = cam1.translation.reshape(3, 1)
    t2 = cam2.translation.reshape(3, 1)

    C1 = -R1.T @ t1
    C2 = -R2.T @ t2

    R_rel = R2 @ R1.T # Relative rotation matrix
    t_rel = C2 - C1  # Relative translation vector

    t_skew = skew_symmetric(t_rel.flatten())

    E = t_skew @ R_rel # Essential matrix

    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1) # Fundamental matrix
    F /= np.linalg.norm(F)

    return F


def compute_all_fundamental_matrices(camera_group):
    """Computes fundamental matrices between all pairs of cameras."""
    cameras = camera_group.cameras
    n_cams = len(cameras)
    F_matrices = {}
    for i in range(n_cams):
        for j in range(i + 1, n_cams):
            F = compute_fundamental_matrix(cameras[i], cameras[j])
            F_matrices[(i, j)] = F
    return F_matrices


class Cluster:
    def __init__(self, node):
        # node: (cam_idx, track_idx)
        self.parent = self
        self.nodes = set([node])
        self.cameras = set([node[0]])

def find(cluster):
    if cluster.parent != cluster:
        cluster.parent = find(cluster.parent)
    return cluster.parent

def union(cluster1, cluster2):
    root1 = find(cluster1)
    root2 = find(cluster2)
    if root1 == root2:
        return False

    if root1.cameras & root2.cameras:
        return False

    root2.parent = root1
    root1.nodes.update(root2.nodes)
    root1.cameras.update(root2.cameras)
    return True


def compute_symmetric_epipolar_distance(p1, p2, F, K1, K2):
    """Computes the symmetric epipolar distance between points in two images.

    Args:
        p1: Points in the first image, shape (n_points, 2).
        p2: Corresponding points in the second image, shape (n_points, 2).
        F: Fundamental matrix from the first to the second image, shape (3, 3).

    Returns:
        distances: Average symmetric epipolar distance.
    """
    # Convert normalized points to pixel coordinates
    p1 = normalized_to_pixel(p1, K1)  # (n_points, 2)
    p2 = normalized_to_pixel(p2, K2)  # (n_points, 2)

    p1_hom = np.hstack([p1, np.ones((p1.shape[0], 1))])  # (n_points, 3)
    p2_hom = np.hstack([p2, np.ones((p2.shape[0], 1))])  # (n_points, 3)

    lines2 = p1_hom @ F.T  # (n_points, 3)
    lines1 = p2_hom @ F    # (n_points, 3)

    numerators2 = np.abs(np.sum(lines2 * p2_hom, axis=1))
    denominators2 = np.sqrt(lines2[:, 0] ** 2 + lines2[:, 1] ** 2)
    distances2 = numerators2 / (denominators2 + 1e-8)

    numerators1 = np.abs(np.sum(lines1 * p1_hom, axis=1))
    denominators1 = np.sqrt(lines1[:, 0] ** 2 + lines1[:, 1] ** 2)
    distances1 = numerators1 / (denominators1 + 1e-8)

    distances = (distances1 + distances2) / 2

    return np.mean(distances)


def normalized_to_pixel(normalized_points: np.ndarray, intrinsic_matrix: np.ndarray) -> np.ndarray:
    """
    Converts normalized coordinates to pixel coordinates.

    Args:
        normalized_points: Array of normalized 2D points, shape (N, 2).
        intrinsic_matrix: Camera intrinsic matrix, shape (3, 3).

    Returns:
        pixel_points: Array of pixel coordinates, shape (N, 2).
    """
    points_homogeneous = np.hstack([normalized_points, np.ones((normalized_points.shape[0], 1))])  # Shape (N, 3)
    
    pixel_points_homogeneous = points_homogeneous @ intrinsic_matrix.T  # Shape (N, 3)
    
    pixel_points = pixel_points_homogeneous[:, :2] / pixel_points_homogeneous[:, 2, np.newaxis]
    
    return pixel_points


def prepare_points_with_score_2d(tracks_all_cameras, final_ids, n_bodyparts, n_cams):
    """
    Prepares the points_with_score_2d array for triangulation.
    """
    unique_ids = set(final_ids.values())
    id_to_idx = {id: idx for idx, id in enumerate(sorted(unique_ids))}
    n_tracks = len(unique_ids)

    points_with_score_2d = np.full((n_cams, n_tracks, n_bodyparts, 3), np.nan)

    for (cam_idx, track_idx), id in final_ids.items():
        id_idx = id_to_idx[id]
        keypoints = tracks_all_cameras[cam_idx][track_idx]  # (n_bodyparts, 2)

        scores = np.ones((n_bodyparts, 1))
        keypoints_with_scores = np.hstack((keypoints, scores))  # (n_bodyparts, 3)
        points_with_score_2d[cam_idx, id_idx] = keypoints_with_scores

    return points_with_score_2d


def get_undistorted_points(camera_group, all_points_with_score_2d_frame: np.ndarray) -> np.ndarray:
    """Undistort the 2D points for all cameras in a frame.

    Args:
        all_points_with_score_2d_frame: (n_cams, n_tracks, n_bodyparts, 3) array for a single frame.

    Returns:
        Undistorted 2D points as a numpy array with shape (n_cams, n_tracks, n_bodyparts, 2).
    """
    n_cams, n_tracks, n_bodyparts, _ = all_points_with_score_2d_frame.shape

    undistorted_points = np.empty((n_cams, n_tracks, n_bodyparts, 2))

    for cam_idx in range(n_cams):
        cam = camera_group.cameras[cam_idx]
        points_2d_cam = all_points_with_score_2d_frame[cam_idx, ..., :2].reshape(-1, 2)
        undistorted_points_cam = cam.undistort_points(points_2d_cam)  # Normalized
        undistorted_points[cam_idx] = undistorted_points_cam.reshape(n_tracks, n_bodyparts, 2)

    return undistorted_points


def process_frame(pair_F_matrices, camera_group, all_points_with_score_2d_frame_undistorted):
    """
    Args:
        all_points_with_score_2d_frame_undistorted: (n_cams, n_tracks, n_bodyparts, (x, y, score)). coordinates are normalized.
    """
    n_cams, n_tracks, n_bodyparts, _ = all_points_with_score_2d_frame_undistorted.shape

    clusters = {}
    for cam_idx in range(n_cams):
        for track_idx in range(n_tracks):
            node = (cam_idx, track_idx)
            clusters[node] = Cluster(node)

    all_matches = []
    for i in range(n_cams):
        for j in range(i + 1, n_cams):
            matches_ij = compute_pairwise_matches(pair_F_matrices, camera_group, all_points_with_score_2d_frame_undistorted, i, j)
            for cost, (idx_i, idx_j) in matches_ij:
                node_i = (i, idx_i)
                node_j = (j, idx_j)
                all_matches.append((cost, node_i, node_j))

    all_matches.sort()

    for cost, node_i, node_j in all_matches:
        cluster_i = clusters[node_i]
        cluster_j = clusters[node_j]
        if find(cluster_i) != find(cluster_j):
            success = union(cluster_i, cluster_j)
            if not success:
                pass  

    root_to_cluster_nodes = {}
    for cluster in clusters.values():
        root = find(cluster)
        if root not in root_to_cluster_nodes:
            root_to_cluster_nodes[root] = set()
        root_to_cluster_nodes[root].update(cluster.nodes)

    sorted_clusters = sorted(root_to_cluster_nodes.values(), key=lambda x: len(x), reverse=True)

    selected_clusters = sorted_clusters[:n_tracks]

    final_ids = {}
    id_to_cluster = {}
    assigned_ids = set() 

    for id_counter, nodes in enumerate(selected_clusters):
        track_idxs = [track_idx for (_, track_idx) in nodes]
        track_idx_counts = {}
        for idx in track_idxs:
            track_idx_counts[idx] = track_idx_counts.get(idx, 0) + 1

        max_count = max(track_idx_counts.values())
        candidate_track_idxs = [idx for idx, count in track_idx_counts.items() if count == max_count]

        available_candidates = [idx for idx in candidate_track_idxs if idx not in assigned_ids]

        if available_candidates:
            assigned_track_idx = available_candidates[0]
        else:
            all_possible_track_idxs = set(range(n_tracks))
            remaining_ids = all_possible_track_idxs - assigned_ids
            if remaining_ids:
                assigned_track_idx = min(remaining_ids)
            else:
                assigned_track_idx = candidate_track_idxs[0]

        assigned_ids.add(assigned_track_idx)
        id_to_cluster[id_counter] = nodes
        for node in nodes:
            final_ids[node] = assigned_track_idx

    selected_nodes = set()
    for nodes in id_to_cluster.values():
        selected_nodes.update(nodes)
    remaining_nodes = set(clusters.keys()) - selected_nodes

    for node in remaining_nodes:
        cam_idx, track_idx = node
        assigned = False
        for shift in range(n_tracks):
            target_id = (track_idx + shift) % n_tracks
            target_cluster_nodes = id_to_cluster.get(target_id, set())
            cluster_cam_idxs = [cam_idx for (cam_idx, _) in target_cluster_nodes]
            if cam_idx not in cluster_cam_idxs:
                majority_track_idx = final_ids[next(iter(target_cluster_nodes))] if target_cluster_nodes else track_idx
                final_ids[node] = majority_track_idx
                target_cluster_nodes.add(node)
                id_to_cluster[target_id] = target_cluster_nodes
                assigned = True
                break
        if not assigned:
            new_id = len(id_to_cluster)
            id_to_cluster[new_id] = set([node])
            final_ids[node] = track_idx

    if len(id_to_cluster) > n_tracks:
        extra_cluster_ids = list(id_to_cluster.keys())[n_tracks:]
        for extra_id in extra_cluster_ids:
            nodes = id_to_cluster.pop(extra_id)
            for id_counter in range(n_tracks):
                target_cluster_nodes = id_to_cluster[id_counter]
                conflict = any(n[0] == node[0] for n in target_cluster_nodes for node in nodes)
                if not conflict:
                    target_cluster_nodes.update(nodes)
                    for node in nodes:
                        final_ids[node] = final_ids[next(iter(target_cluster_nodes))]
                    break
    elif len(id_to_cluster) < n_tracks:
        for _ in range(n_tracks - len(id_to_cluster)):
            new_id = len(id_to_cluster)
            id_to_cluster[new_id] = set()

    return final_ids
    

def compute_pairwise_matches(pair_F_matrices, camera_group, all_points_with_score_2d_frame_undistorted, cam_idx1, cam_idx2):
    n_cams, n_tracks, n_bodyparts, _ = all_points_with_score_2d_frame_undistorted.shape

    F = pair_F_matrices.get((cam_idx1, cam_idx2))
    K1 = camera_group.cameras[cam_idx1].matrix
    K2 = camera_group.cameras[cam_idx2].matrix

    matches = []
    cost_matrix = np.zeros((n_tracks, n_tracks))

    first_selected_indices = [0, 1, 2] # Only use head, leftear, rightear to compute cost, since they are most reliable
    first_index_mask = np.zeros(n_bodyparts, dtype=bool)
    first_index_mask[first_selected_indices] = True

    second_selected_indices = [3, 8] # Use all other bodyparts to compute cost
    second_index_mask = np.zeros(n_bodyparts, dtype=bool)
    second_index_mask[second_selected_indices] = True

    for track_idx_1, track_points_1 in enumerate(all_points_with_score_2d_frame_undistorted[cam_idx1]):
        for track_idx_2, track_points_2 in enumerate(all_points_with_score_2d_frame_undistorted[cam_idx2]):
            valid_mask = ~np.isnan(track_points_1[:, 0]) & ~np.isnan(track_points_2[:, 0]) & first_index_mask
            if np.sum(valid_mask) == 0:
                valid_mask = ~np.isnan(track_points_1[:, 0]) & ~np.isnan(track_points_2[:, 0]) & second_index_mask

            if np.sum(valid_mask) == 0:
                cost = 50000  # Set to a large value
            else:
                distances = compute_symmetric_epipolar_distance(track_points_1[valid_mask], track_points_2[valid_mask], F, K1, K2)
                cost = np.mean(distances)

            cost_matrix[track_idx_1, track_idx_2] = cost
            matches.append((cost, (track_idx_1, track_idx_2)))
    return matches
