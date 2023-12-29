import cv2
import json
import queue
import numpy as np
from tqdm import trange
from pprint import pprint
from scipy import optimize
from scipy.sparse import dok_matrix
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.vq import whiten
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any

from marmopose.calibration.boards import merge_rows, extract_points, extract_rtvecs
from marmopose.utils.helpers import get_video_params


def triangulate_SVD(points: List[Tuple[float, float]], camera_mats: List[np.ndarray]) -> np.ndarray:
    num_cams = len(camera_mats)
    A = np.zeros((num_cams * 2, 4))
    for i in range(num_cams):
        x, y = points[i]
        mat = camera_mats[i]
        A[(i * 2):(i * 2 + 1)] = x * mat[2] - mat[0]
        A[(i * 2 + 1):(i * 2 + 2)] = y * mat[2] - mat[1]
    _, _, vh = np.linalg.svd(A, full_matrices=True)
    p3d = vh[-1]
    p3d = p3d[:3] / p3d[3]
    return p3d


class Camera:
    def __init__(self,
                 name: str = None,
                 size: Tuple[int, int] = None,
                 matrix: np.ndarray = np.eye(3),
                 distortion: np.ndarray = np.zeros(5),
                 rotation: np.ndarray = np.zeros(3),
                 translation: np.ndarray = np.zeros(3),
                 extra_distortion: bool = True):
        self.set_name(name)
        self.set_size(size)
        self.set_camera_matrix(matrix)
        self.set_distortion(distortion)
        self.set_rotation(rotation)
        self.set_translation(translation)
        self.extra_distortion = extra_distortion
    
    def load_dict(self, d: dict) -> None:
        self.set_name(d['name'])
        self.set_size(d['size'])
        self.set_camera_matrix(d['matrix'])
        self.set_distortion(d['distortion'])
        self.set_rotation(d['rotation'])
        self.set_translation(d['translation'])
    
    @classmethod
    def from_dict(cls, d: dict) -> 'Camera':
        cam = cls()
        cam.load_dict(d)
        return cam
    
    def get_dict(self) -> Dict:
        return {
            'name': self.get_name(),
            'size': list(self.get_size()),
            'matrix': self.get_camera_matrix().tolist(),
            'distortion': self.get_distortion().tolist(),
            'rotation': self.get_rotation().tolist(),
            'translation': self.get_translation().tolist(),
        }
    
    def set_camera_matrix(self, matrix):
        self.matrix = np.array(matrix)
    
    def get_camera_matrix(self):
        return self.matrix

    def set_distortion(self, distortion):
        self.distortion = np.array(distortion).ravel()
    
    def get_distortion(self):
        return self.distortion

    def set_rotation(self, rotation):
        self.rotation = np.array(rotation).ravel()
    
    def get_rotation(self):
        return self.rotation
    
    def set_translation(self, translation):
        self.translation = np.array(translation).ravel()
    
    def get_translation(self):
        return self.translation
    
    def set_name(self, name: str):
        self.name = name
    
    def get_name(self):
        return self.name
    
    def set_size(self, size: Tuple[int, int]):
        self.size = size
    
    def get_size(self) -> Tuple[int, int]:
        return self.size
    
    def set_focal_length(self, fx: float, fy: float = None):
        if fy is None:
            fy = fx
        self.matrix[0, 0] = fx
        self.matrix[1, 1] = fy

    def get_focal_length(self, both: bool = False):
        fx = self.matrix[0, 0]
        fy = self.matrix[1, 1]
        if both:
            return (fx, fy)
        else:
            return (fx + fy) / 2.0
    
    def get_extrinsic_matrix(self):
        return get_extrinsic_matrix(self.rotation, self.translation)
    
    def set_params(self, params):
        self.set_rotation(params[0:3])
        self.set_translation(params[3:6])
        self.set_focal_length(params[6])

        distortion = np.zeros(5)
        distortion[0] = params[7]
        if self.extra_distortion:
            distortion[1] = params[8]
        self.set_distortion(distortion)

    def get_params(self):
        params = np.zeros(8 + self.extra_distortion)
        params[0:3] = self.get_rotation()
        params[3:6] = self.get_translation()
        params[6] = self.get_focal_length()
        distortion = self.get_distortion()
        params[7] = distortion[0]
        if self.extra_distortion:
            params[8] = distortion[1]
        return params
    
    def resize_camera(self, scale: float) -> None:
        size = self.get_size()
        new_size = (size[0] * scale, size[1] * scale)
        matrix = self.get_camera_matrix()
        new_matrix = matrix * scale
        new_matrix[2, 2] = 1
        self.set_size(new_size)
        self.set_camera_matrix(new_matrix)
    
    def distort_points(self, points_2d: np.ndarray) -> np.ndarray:
        """Apply lens distortion to a set of 2D points using pinhole camera model.

        Args:
            points_2d: Input points in 2D, shape (N, 2).

        Returns:
            Distorted points in 2D, shape (N, 2).
        """
        reshaped_points = points_2d.reshape(-1, 1, 2)
        homogeneous_points = np.dstack([reshaped_points, np.ones(points_2d.shape[0], 1, 1)])
        distorted_points, _ = cv2.projectPoints(homogeneous_points, np.zeros(3), np.zeros(3), self.matrix, self.distortion)
        return distorted_points.reshape(points_2d.shape)
    
    def undistort_points(self, points_2d: np.ndarray) -> np.ndarray:
        """Remove lens distortion from a set of 2D points using pinhole camera model.

        Args:
            points_2d: Distorted points in 2D, shape (N, 2).

        Returns:
            Undistorted points in 2D, shape (N, 2).
        """
        reshaped_points = points_2d.reshape(-1, 1, 2)
        undistorted_points = cv2.undistortPoints(reshaped_points, self.matrix, self.distortion)
        return undistorted_points.reshape(points_2d.shape)

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D image plane using the camera's intrinsic and extrinsic parameters.

        Args:
            points_3d: Points in 3D space, shape (N, 3).

        Returns:
            Points projected to 2D, shape (N, 2).
        """
        reshaped_points = points_3d.reshape(-1, 1, 3)
        projected_points, _ = cv2.projectPoints(reshaped_points, self.rotation, self.translation, self.matrix, self.distortion)
        reshaped_projected_points = projected_points.reshape(-1, 2)
        return reshaped_projected_points

    def reprojection_error(self, points_3d: np.ndarray, points_2d: np.ndarray) -> np.ndarray:
        """Calculates the reprojection error between 3D points and their corresponding 2D points in the image plane.

        Args:
            points_3d: Array of shape (N, 3) containing the 3D coordinates of N points.
            points_2d: Array of shape (N, 2) containing the 2D coordinates of N points in the image plane.

        Returns:
            Array of shape (N, 2) containing the absolute difference between the projected 2D points and the actual 2D points.
        """
        projected_points = self.project(points_3d)
        return np.abs(points_2d - projected_points)
    
    def copy(self) -> 'Camera':
        """Returns a copy of the current Camera object."""
        return Camera(name = self.get_name(), 
                      size = self.get_size(),
                      matrix = self.get_camera_matrix().copy(),
                      distortion = self.get_distortion().copy(),
                      rotation = self.get_rotation().copy(),
                      translation = self.get_translation().copy(),
                      extra_distortion = self.extra_distortion)


class FisheyeCamera(Camera):
    def __init__(self,
                 name: str = None,
                 size: Tuple[int, int] = None,
                 matrix: np.ndarray = np.eye(3),
                 distortion: np.ndarray = np.zeros(4),
                 rotation: np.ndarray = np.zeros(3),
                 translation: np.ndarray = np.zeros(3),
                 extra_distortion: bool = False):
        super().__init__(name, size, matrix, distortion, rotation, translation, extra_distortion)

    def get_dict(self) -> Dict[str, Any]:
        d = super().get_dict()
        d['fisheye'] = True
        return d
    
    def set_params(self, params):
        self.set_rotation(params[0:3])
        self.set_translation(params[3:6])
        self.set_focal_length(params[6])

        distortion = np.zeros(4)
        distortion[0] = params[7]
        if self.extra_distortion:
            distortion[1] = params[8]
        self.set_distortion(distortion)

    def get_params(self):
        params = np.zeros(8+self.extra_distortion)
        params[0:3] = self.get_rotation()
        params[3:6] = self.get_translation()
        params[6] = self.get_focal_length()
        dist = self.get_distortion()
        params[7] = dist[0]
        if self.extra_distortion:
            params[8] = dist[1]
        return params
    

    def distort_points(self, points_2d: np.ndarray) -> np.ndarray:
        """Apply lens distortion to a set of 2D points using fisheye camera model.

        Args:
            points_2d: The input points in a NumPy array of shape (N, 2).

        Returns:
            The distorted points in a NumPy array of shape (N, 2).
        """
        reshaped_points = points_2d.reshape(-1, 1, 2)
        homogeneous_points = np.dstack([reshaped_points, np.ones((reshaped_points.shape[0], 1, 1))])
        distorted_points, _ = cv2.fisheye.projectPoints(homogeneous_points, np.zeros(3), np.zeros(3), self.matrix, self.distortion)
        return distorted_points.reshape(points_2d.shape)

    def undistort_points(self, points_2d: np.ndarray) -> np.ndarray:
        """Remove lens distortion from a set of 2D points using fisheye camera model.

        Args:
            points_2d: The input distorted points in a NumPy array of shape (N, 2).

        Returns:
            The undistorted points in a NumPy array of shape (N, 2).
        """
        reshaped_points = points_2d.reshape(-1, 1, 2)
        undistorted_points = cv2.fisheye.undistortPoints(reshaped_points, self.matrix, self.distortion)
        return undistorted_points.reshape(points_2d.shape)

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D image plane using the camera's intrinsic and extrinsic parameters.

        Args:
            points: The input 3D points in a NumPy array of shape (N, 3).

        Returns:
            The projected 2D points in a NumPy array of shape (N, 1, 2).
        """
        reshaped_points = points_3d.reshape(-1, 1, 3)
        projected_points, _ = cv2.fisheye.projectPoints(reshaped_points, self.rotation, self.translation, self.matrix, self.distortion)
        reshaped_projected_points = projected_points.reshape(-1, 2)
        return reshaped_projected_points
    
    def copy(self) -> 'FisheyeCamera':
        """Returns a copy of the current FisheyeCamera object."""
        return FisheyeCamera(name = self.get_name(), 
                     size = self.get_size(), 
                     matrix = self.get_camera_matrix().copy(), 
                     distortion = self.get_distortion().copy(), 
                     rotation = self.get_rotation().copy(), 
                     translation = self.get_translation().copy(), 
                     extra_distortion = self.extra_distortion)


class CameraGroup:
    def __init__(self, cameras: List, metadata: Dict = {}):
        self.cameras = cameras
        self.metadata = metadata

    def get_names(self) -> List[str]:
        return [cam.get_name() for cam in self.cameras]
    
    def get_dicts(self):
        return [cam.get_dict() for cam in self.cameras]
    
    def get_rotations(self):
        return np.array([cam.get_rotation() for cam in self.cameras])

    def get_translations(self):
        return np.array([cam.get_translation() for cam in self.cameras])
    
    def set_rotations(self, rotations):
        for cam, rotation in zip(self.cameras, rotations):
            cam.set_rotation(rotation)

    def set_translations(self, translations):
        for cam, translation in zip(self.cameras, translations):
            cam.set_translation(translation)

    def resize_cameras(self, scale: float) -> None:
        for cam in self.cameras:
            cam.resize_camera(scale)
    
    @classmethod
    def from_names(cls, names, fisheye=False):
        cameras = [FisheyeCamera(name=name) if fisheye else Camera(name=name) for name in names]
        return cls(cameras)
    
    @classmethod
    def from_dicts(cls, dicts: List[Dict]) -> 'CameraGroup':
        cameras = [FisheyeCamera.from_dict(d) if d.get('fisheye') else Camera.from_dict(d) for d in dicts]
        return cls(cameras)
    
    @staticmethod
    def load_from_json(file_path: str) -> 'CameraGroup':
        with open(file_path, 'r') as f:
            params_dict = json.load(f)

        items = [v for k, v in sorted(params_dict.items()) if k != 'metadata']
        camera_group = CameraGroup.from_dicts(items)
        camera_group.metadata = params_dict.get('metadata', {})
        return camera_group

    def save_to_json(self, file_path: str):
        dicts = self.get_dicts()
        names = [d['name'] for d in dicts]
        params_dict = dict(zip(names, dicts))
        params_dict['metadata'] = self.metadata

        with open(file_path, 'w') as f:
            json.dump(params_dict, f, indent=4)

    def subset_cameras(self, indices: List[int]) -> 'CameraGroup':
        return CameraGroup([self.cameras[ix].copy() for ix in indices], self.metadata)

    def subset_cameras_names(self, names: List[str]) -> 'CameraGroup':
        cur_names_dict = {name: idx for idx, name in enumerate(self.get_names())}
        indices = [cur_names_dict[name] for name in names if name in cur_names_dict]
        if len(names) != len(indices):
            missing_names = set(names) - set(cur_names_dict.keys())
            raise IndexError(f"names {missing_names} not part of camera names: {list(cur_names_dict.keys())}")
        return self.subset_cameras(indices)
    
    def set_camera_sizes_videos(self, videos):
        for cam, cam_videos in zip(self.cameras, videos):
            for vidname in cam_videos:
                params = get_video_params(vidname)
                size = (params['width'], params['height'])
                cam.set_size(size)
    
    def triangulate(self, points_2d_flat: np.ndarray, undistort: bool = True, verbose: bool = False) -> np.ndarray:
        """Triangulate 3D points from 2D points using camera extrinsic matrices.

        Args:
            points_2d_flat: 2D points of shape (n_cameras, n_points, 2).
            undistort (optional): Whether to undistort the 2D points using camera intrinsic matrices. Defaults to True.
            verbose (optional): Whether to display a progress bar. Defaults to False.

        Returns:
            3D points of shape (n_points, 3).
        """
        if undistort:
            points_2d_flat = np.array([cam.undistort_points(np.copy(pt)) for pt, cam in zip(points_2d_flat, self.cameras)])
        cam_mats = np.array([cam.get_extrinsic_matrix() for cam in self.cameras])

        n_points = points_2d_flat.shape[1]
        points_3d_flat = np.full((n_points, 3), np.nan)

        iterator = trange(n_points, ncols=100, desc='Triangulating... ', unit='points') if verbose else range(n_points)
        for pt_idx in iterator:
            sub_points = points_2d_flat[:, pt_idx, :]
            valid_points = ~np.isnan(sub_points[:, 0])
            if np.sum(valid_points) >= 2:
                points_3d_flat[pt_idx] = triangulate_SVD(sub_points[valid_points], cam_mats[valid_points])

        return points_3d_flat
    
    def reproject(self, points_3d):
        """Given an Nx3 array of 3D points, projects the points to 2D image plane.

        Args:
            points_3d: 3D points with shape (N, 3).

        Returns:
            2D points with shape (C, N, 2).
        """
        points_2d_reprojected = np.array([cam.project(points_3d) for cam in self.cameras])
        return points_2d_reprojected

    def reprojection_error(self, points_3d: np.ndarray, points_2d: np.ndarray, mean: bool = False) -> np.ndarray:
        """Compute the reprojection error between observed 2D points and reprojected 2D points from 3D coordinates.
        
        Args:
            points_3d: 3D coordinates, shape (N, 3).
            points_2d: Observed 2D coordinates from cameras, shape (C, N, 2).
            mean: If True, return the mean error across all cameras for each point.
            
        Returns:
            errors: Reprojection error. If mean is True, returns mean error for each point, shape (N,).
                    Otherwise, returns error for each camera and each point, shape (C, N, 2).
        """
        points_2d_reprojected = self.reproject(points_3d)
        errors = np.abs(points_2d - points_2d_reprojected)

        if mean:
            errors_norm = np.linalg.norm(errors, axis=2)
            good = ~np.isnan(errors_norm)
            errors_norm[~good] = 0
            denom = np.sum(good, axis=0).astype('float64')
            denom[denom < 2] = np.nan
            errors = np.sum(errors_norm, axis=0) / denom

        return errors

    

    # TODO: Refractor the following functions
    def calibrate_rows(self, all_rows, board, 
                       init_intrinsics=True, init_extrinsics=True, verbose=True, **kwargs):
        assert len(all_rows) == len(self.cameras), "Number of camera detections does not match number of cameras"

        for rows, camera in zip(all_rows, self.cameras):
            size = camera.get_size()

            assert size is not None, f"Camera with name {camera.get_name()} has no specified frame size"

            if init_intrinsics:
                objp, imgp = board.get_all_calibration_points(rows)
                mixed = [(o, i) for (o, i) in zip(objp, imgp) if len(o) >= 7]
                objp, imgp = zip(*mixed)
                matrix = cv2.initCameraMatrix2D(objp, imgp, tuple(size))


                # ======================================================================
                # TODO: Optimize this, it is used to avoid nan in init camera matrix, better calibration video could avoid this
                # matrix[0][0] = matrix[1][1] = 1078.0
                # matrix = np.nan_to_num(matrix, nan=1000.0)
                print(matrix)
                # ======================================================================

                camera.set_camera_matrix(matrix)

        for i, (row, cam) in enumerate(zip(all_rows, self.cameras)):
            all_rows[i] = board.estimate_pose_rows(cam, row)

        merged = merge_rows(all_rows)
        imgp, extra = extract_points(merged, board, min_cameras=2)

        if init_extrinsics:
            rtvecs = extract_rtvecs(merged)
            if verbose: pprint(get_connections(rtvecs, self.get_names()))
            rvecs, tvecs = get_initial_extrinsics(rtvecs, self.get_names())
            self.set_rotations(rvecs)
            self.set_translations(tvecs)

        error = self.bundle_adjust_iter(imgp, extra, verbose=verbose, **kwargs)
        self.metadata['error'] = error
    
    def average_error(self, p2ds, median=False):
        p3ds = self.triangulate(p2ds)
        errors = self.reprojection_error(p3ds, p2ds, mean=True)
        if median:
            return np.median(errors)
        else:
            return np.mean(errors)

    def bundle_adjust_iter(self, p2ds, extra=None,
                           n_iters=10, start_mu=15, end_mu=1,
                           max_nfev=200, ftol=1e-4,
                           n_samp_iter=100, n_samp_full=1000,
                           error_threshold=0.3,
                           verbose=False):
        """Given an CxNx2 array of 2D points,
        where N is the number of points and C is the number of cameras,
        this performs iterative bundle adjustsment to fine-tune the parameters of the cameras.
        That is, it performs bundle adjustment multiple times, adjusting the weights given to points
        to reduce the influence of outliers.
        This is inspired by the algorithm for Fast Global Registration by Zhou, Park, and Koltun
        """

        assert p2ds.shape[0] == len(self.cameras), \
            "Invalid points shape, first dim should be equal to" \
            " number of cameras ({}), but shape is {}".format(
                len(self.cameras), p2ds.shape
            )

        p2ds_full = p2ds
        extra_full = extra

        p2ds, extra = resample_points(p2ds_full, extra_full,
                                      n_samp=n_samp_full)
        error = self.average_error(p2ds, median=True)

        if verbose:
            print('error: ', error)

        mus = np.exp(np.linspace(np.log(start_mu), np.log(end_mu), num=n_iters))

        if verbose:
            print('n_samples: {}'.format(n_samp_iter))

        for i in range(n_iters):
            p2ds, extra = resample_points(p2ds_full, extra_full,
                                          n_samp=n_samp_full)
            p3ds = self.triangulate(p2ds)
            errors_full = self.reprojection_error(p3ds, p2ds, mean=False)
            errors_norm = self.reprojection_error(p3ds, p2ds, mean=True)

            error_dict = get_error_dict(errors_full)
            max_error = 0
            min_error = 0
            for k, v in error_dict.items():
                num, percents = v
                max_error = max(percents[-1], max_error)
                min_error = max(percents[0], min_error)
            mu = max(min(max_error, mus[i]), min_error)

            good = errors_norm < mu
            extra_good = subset_extra(extra, good)
            p2ds_samp, extra_samp = resample_points(
                p2ds[:, good], extra_good, n_samp=n_samp_iter)

            error = np.median(errors_norm)

            if error < error_threshold:
                break

            if verbose:
                pprint(error_dict)
                print('error: {:.2f}, mu: {:.1f}, ratio: {:.3f}'.format(error, mu, np.mean(good)))

            self.bundle_adjust(p2ds_samp, extra_samp,
                               loss='linear', ftol=ftol,
                               max_nfev=max_nfev,
                               verbose=verbose)

        p2ds, extra = resample_points(p2ds_full, extra_full,
                                      n_samp=n_samp_full)
        p3ds = self.triangulate(p2ds)
        errors_full = self.reprojection_error(p3ds, p2ds, mean=False)
        errors_norm = self.reprojection_error(p3ds, p2ds, mean=True)
        error_dict = get_error_dict(errors_full)
        if verbose:
            pprint(error_dict)

        max_error = 0
        min_error = 0
        for k, v in error_dict.items():
            num, percents = v
            max_error = max(percents[-1], max_error)
            min_error = max(percents[0], min_error)
        mu = max(max(max_error, end_mu), min_error)

        good = errors_norm < mu
        extra_good = subset_extra(extra, good)
        self.bundle_adjust(p2ds[:, good], extra_good,
                           loss='linear',
                           ftol=ftol, max_nfev=max(200, max_nfev),
                           verbose=verbose)

        error = self.average_error(p2ds, median=True)

        p3ds = self.triangulate(p2ds)
        errors_full = self.reprojection_error(p3ds, p2ds, mean=False)
        error_dict = get_error_dict(errors_full)
        if verbose:
            pprint(error_dict)

        if verbose:
            print('error: ', error)

        return error
    

    def bundle_adjust(self, p2ds, extra=None,
                      loss='linear',
                      threshold=50,
                      ftol=1e-4,
                      max_nfev=1000,
                      weights=None,
                      start_params=None,
                      verbose=True):
        """Given an CxNx2 array of 2D points,
        where N is the number of points and C is the number of cameras,
        this performs bundle adjustsment to fine-tune the parameters of the cameras"""

        assert p2ds.shape[0] == len(self.cameras), \
            "Invalid points shape, first dim should be equal to" \
            " number of cameras ({}), but shape is {}".format(
                len(self.cameras), p2ds.shape
            )

        if extra is not None:
            extra['ids_map'] = remap_ids(extra['ids'])

        x0, n_cam_params = self._initialize_params_bundle(p2ds, extra)

        if start_params is not None:
            x0 = start_params
            n_cam_params = len(self.cameras[0].get_params())

        error_fun = self._error_fun_bundle

        jac_sparse = self._jac_sparsity_bundle(p2ds, n_cam_params, extra)

        f_scale = threshold
        opt = optimize.least_squares(error_fun,
                                     x0,
                                     jac_sparsity=jac_sparse,
                                     f_scale=f_scale,
                                     x_scale='jac',
                                     loss=loss,
                                     ftol=ftol,
                                     method='trf',
                                     tr_solver='lsmr',
                                     verbose=2 * verbose,
                                     max_nfev=max_nfev,
                                     args=(p2ds, n_cam_params, extra))
        best_params = opt.x

        for i, cam in enumerate(self.cameras):
            a = i * n_cam_params
            b = (i + 1) * n_cam_params
            cam.set_params(best_params[a:b])

        error = self.average_error(p2ds)
        return error
    
    def _error_fun_bundle(self, params, p2ds, n_cam_params, extra):
        """Error function for bundle adjustment"""
        good = ~np.isnan(p2ds)
        n_cams = len(self.cameras)

        for i in range(n_cams):
            cam = self.cameras[i]
            a = i * n_cam_params
            b = (i + 1) * n_cam_params
            cam.set_params(params[a:b])

        n_cams = len(self.cameras)
        sub = n_cam_params * n_cams
        n3d = p2ds.shape[1] * 3
        p3ds_test = params[sub:sub+n3d].reshape(-1, 3)
        errors = self.reprojection_error(p3ds_test, p2ds)
        errors_reproj = errors[good]

        if extra is not None:
            ids = extra['ids_map']
            objp = extra['objp']
            min_scale = np.min(objp[objp > 0])
            n_boards = int(np.max(ids)) + 1
            a = sub+n3d
            rvecs = params[a:a+n_boards*3].reshape(-1, 3)
            tvecs = params[a+n_boards*3:a+n_boards*6].reshape(-1, 3)
            expected = transform_points(objp, rvecs[ids], tvecs[ids])
            errors_obj = 2 * (p3ds_test - expected).ravel() / min_scale
        else:
            errors_obj = np.array([])

        return np.hstack([errors_reproj, errors_obj])


    def _jac_sparsity_bundle(self, p2ds, n_cam_params, extra):
        """Given an CxNx2 array of 2D points,
        where N is the number of points and C is the number of cameras,
        compute the sparsity structure of the jacobian for bundle adjustment"""

        point_indices = np.zeros(p2ds.shape, dtype='int32')
        cam_indices = np.zeros(p2ds.shape, dtype='int32')

        for i in range(p2ds.shape[1]):
            point_indices[:, i] = i

        for j in range(p2ds.shape[0]):
            cam_indices[j] = j

        good = ~np.isnan(p2ds)

        if extra is not None:
            ids = extra['ids_map']
            n_boards = int(np.max(ids)) + 1
            total_board_params = n_boards * (3 + 3) # rvecs + tvecs
        else:
            n_boards = 0
            total_board_params = 0

        n_cams = p2ds.shape[0]
        n_points = p2ds.shape[1]
        total_params_reproj = n_cams * n_cam_params + n_points * 3
        n_params = total_params_reproj + total_board_params

        n_good_values = np.sum(good)
        if extra is not None:
            n_errors = n_good_values + n_points * 3
        else:
            n_errors = n_good_values

        A_sparse = dok_matrix((n_errors, n_params), dtype='int16')

        cam_indices_good = cam_indices[good]
        point_indices_good = point_indices[good]

        # -- reprojection error --
        ix = np.arange(n_good_values)

        ## update camera params based on point error
        for i in range(n_cam_params):
            A_sparse[ix, cam_indices_good * n_cam_params + i] = 1

        ## update point position based on point error
        for i in range(3):
            A_sparse[ix, n_cams * n_cam_params + point_indices_good * 3 + i] = 1

        # -- match for the object points--
        if extra is not None:
            point_ix = np.arange(n_points)

            ## update all the camera parameters
            # A_sparse[n_good_values:n_good_values+n_points*3,
            #          0:n_cams*n_cam_params] = 1

            ## update board rotation and translation based on error from expected
            for i in range(3):
                for j in range(3):
                    A_sparse[n_good_values + point_ix*3 + i,
                             total_params_reproj + ids*3 + j] = 1
                    A_sparse[n_good_values + point_ix*3 + i,
                             total_params_reproj + n_boards*3 + ids*3 + j] = 1


            ## update point position based on error from expected
            for i in range(3):
                A_sparse[n_good_values + point_ix*3 + i,
                         n_cams*n_cam_params + point_ix*3 + i] = 1


        return A_sparse

    def _initialize_params_bundle(self, p2ds, extra):
        """Given an CxNx2 array of 2D points,
        where N is the number of points and C is the number of cameras,
        initializes the parameters for bundle adjustment"""

        cam_params = np.hstack([cam.get_params() for cam in self.cameras])
        n_cam_params = len(cam_params) // len(self.cameras)

        total_cam_params = len(cam_params)

        n_cams, n_points, _ = p2ds.shape
        assert n_cams == len(self.cameras), \
            "number of cameras in CameraGroup does not " \
            "match number of cameras in 2D points given"

        p3ds = self.triangulate(p2ds)

        if extra is not None:
            ids = extra['ids_map']
            n_boards = int(np.max(ids[~np.isnan(ids)])) + 1
            total_board_params = n_boards * (3 + 3) # rvecs + tvecs

            # initialize to 0
            rvecs = np.zeros((n_boards, 3), dtype='float64')
            tvecs = np.zeros((n_boards, 3), dtype='float64')

            if 'rvecs' in extra and 'tvecs' in extra:
                rvecs_all = extra['rvecs']
                tvecs_all = extra['tvecs']
                for board_num in range(n_boards):
                    point_id = np.where(ids == board_num)[0][0]
                    cam_ids_possible = np.where(~np.isnan(p2ds[:, point_id, 0]))[0]
                    cam_id = np.random.choice(cam_ids_possible)
                    M_cam = self.cameras[cam_id].get_extrinsic_matrix()
                    M_board_cam = get_extrinsic_matrix(rvecs_all[cam_id, point_id],
                                         tvecs_all[cam_id, point_id])
                    M_board = np.matmul(np.linalg.inv(M_cam), M_board_cam)
                    rvec, tvec = get_rtvec(M_board)
                    rvecs[board_num] = rvec
                    tvecs[board_num] = tvec


        else:
            total_board_params = 0

        x0 = np.zeros(total_cam_params + p3ds.size + total_board_params)
        x0[:total_cam_params] = cam_params
        x0[total_cam_params:total_cam_params+p3ds.size] = p3ds.ravel()

        if extra is not None:
            start_board = total_cam_params+p3ds.size
            x0[start_board:start_board + n_boards*3] = rvecs.ravel()
            x0[start_board + n_boards*3:start_board + n_boards*6] = \
                tvecs.ravel()

        return x0, n_cam_params


def get_extrinsic_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """
    Constructs a transformation matrix given a rotation and translation vector.

    Args:
        rvec: Rotation vector.
        tvec: Translation vector.

    Returns:
        The 4x4 transformation matrix.
    """
    rotmat, _ = cv2.Rodrigues(rvec)
    M = np.eye(4)
    M[:3, :3] = rotmat
    M[:3, 3] = tvec.flatten()
    return M


# TODO: Refractor these functions, move them to appropriate modules
def get_connections(xs, cam_names=None, both=True):
    n_cams = xs.shape[0]
    n_points = xs.shape[1]

    if cam_names is None:
        cam_names = np.arange(n_cams)

    connections = defaultdict(int)

    for rnum in range(n_points):
        ixs = np.where(~np.isnan(xs[:, rnum, 0]))[0]
        keys = [cam_names[ix] for ix in ixs]
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                a = keys[i]
                b = keys[j]
                connections[(a,b)] += 1
                if both:
                    connections[(b,a)] += 1

    return connections


def transform_points(points, rvecs, tvecs):
    """Rotate points by given rotation vectors and translate.
    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rvecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rvecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    rotated = cos_theta * points + \
        sin_theta * np.cross(v, points) + \
        dot * (1 - cos_theta) * v

    return rotated + tvecs


def remap_ids(ids):
    unique_ids = np.unique(ids)
    ids_out = np.copy(ids)
    for i, num in enumerate(unique_ids):
        ids_out[ids == num] = i
    return ids_out


def get_error_dict(errors_full, min_points=10):
    n_cams = errors_full.shape[0]
    errors_norm = np.linalg.norm(errors_full, axis=2)

    good = ~np.isnan(errors_full[:, :, 0])

    error_dict = dict()

    for i in range(n_cams):
        for j in range(i+1, n_cams):
            subset = good[i] & good[j]
            err_subset = errors_norm[:, subset][[i, j]]
            err_subset_mean = np.mean(err_subset, axis=0)
            if np.sum(subset) > min_points:
                percents = np.percentile(err_subset_mean, [15, 75])
                # percents = np.percentile(err_subset, [25, 75])
                error_dict[(i, j)] = (err_subset.shape[1], percents)
    return error_dict

def resample_points(imgp, extra=None, n_samp=25):
    # if extra is not None:
    #     return resample_points_extra(imgp, extra, n_samp)

    n_cams = imgp.shape[0]
    good = ~np.isnan(imgp[:, :, 0])
    ixs = np.arange(imgp.shape[1])

    num_cams = np.sum(~np.isnan(imgp[:, :, 0]), axis=0)

    include = set()

    for i in range(n_cams):
        for j in range(i+1, n_cams):
            subset = good[i] & good[j]
            n_good = np.sum(subset)
            if n_good > 0:
                ## pick points, prioritizing points seen by more cameras
                arr = np.copy(num_cams[subset]).astype('float64')
                arr += np.random.random(size=arr.shape)
                picked_ix = np.argsort(-arr)[:n_samp]
                picked = ixs[subset][picked_ix]
                include.update(picked)

    final_ixs = sorted(include)
    newp = imgp[:, final_ixs]
    extra = subset_extra(extra, final_ixs)
    return newp, extra

def subset_extra(extra, ixs):
    if extra is None:
        return None

    new_extra = {
        'objp': extra['objp'][ixs],
        'ids': extra['ids'][ixs],
        'rvecs': extra['rvecs'][:, ixs],
        'tvecs': extra['tvecs'][:, ixs]
    }
    return new_extra


def get_initial_extrinsics(rtvecs, cam_names=None):
    graph = get_calibration_graph(rtvecs, cam_names)
    pairs = find_calibration_pairs(graph, source=0)
    extrinsics = compute_camera_matrices(rtvecs, pairs)

    n_cams = rtvecs.shape[0]
    rvecs = []
    tvecs = []
    for cnum in range(n_cams):
        rvec, tvec = get_rtvec(extrinsics[cnum])
        rvecs.append(rvec)
        tvecs.append(tvec)
    rvecs = np.array(rvecs)
    tvecs = np.array(tvecs)
    return rvecs, tvecs


def get_calibration_graph(rtvecs, cam_names=None):
    n_cams = rtvecs.shape[0]
    n_points = rtvecs.shape[1]

    if cam_names is None:
        cam_names = np.arange(n_cams)

    connections = get_connections(rtvecs, np.arange(n_cams))

    components = dict(zip(np.arange(n_cams), range(n_cams)))
    edges = set(connections.items())

    graph = defaultdict(list)

    for edgenum in range(n_cams-1):
        if len(edges) == 0:
            component_names = dict()
            for k,v in list(components.items()):
                component_names[cam_names[k]] = v
            raise ValueError("""
Could not build calibration graph.
Some group of cameras could not be paired by simultaneous calibration board detections.
Check which cameras have different group numbers below to see the missing edges.
{}""".format(component_names))

        (a, b), weight = max(edges, key=lambda x: x[1])
        graph[a].append(b)
        graph[b].append(a)

        match = components[a]
        replace = components[b]
        for k, v in components.items():
            if match == v:
                components[k] = replace

        for e in edges.copy():
            (a,b), w = e
            if components[a] == components[b]:
                edges.remove(e)

    return graph


def find_calibration_pairs(graph, source=None):
    pairs = []
    explored = set()

    if source is None:
        source = sorted(graph.keys())[0]

    q = queue.deque()
    q.append(source)

    while len(q) > 0:
        item = q.pop()
        explored.add(item)

        for new in graph[item]:
            if new not in explored:
                q.append(new)
                pairs.append( (item, new) )
    return pairs


def compute_camera_matrices(rtvecs, pairs):
    extrinsics = dict()
    source = pairs[0][0]
    extrinsics[source] = np.identity(4)
    for (a,b) in pairs:
        ext = get_transform(rtvecs, b, a)
        extrinsics[b] = np.matmul(ext, extrinsics[a])
    return extrinsics


def get_transform(rtvecs, left, right):
    L = []
    for dix in range(rtvecs.shape[1]):
        d = rtvecs[:, dix]
        good = ~np.isnan(d[:, 0])

        if good[left] and good[right]:
            M_left = get_extrinsic_matrix(d[left, 0:3], d[left, 3:6])
            M_right = get_extrinsic_matrix(d[right, 0:3], d[right, 3:6])
            M = np.matmul(M_left, np.linalg.inv(M_right))
            L.append(M)
    L_best = select_matrices(L)
    M_mean = mean_transform(L_best)
    # M_mean = mean_transform_robust(L, M_mean, error=0.5)
    # M_mean = mean_transform_robust(L, M_mean, error=0.2)
    M_mean = mean_transform_robust(L, M_mean, error=0.1)
    return M_mean


def select_matrices(Ms):
    Ms = np.array(Ms)
    rvecs = [cv2.Rodrigues(M[:3,:3])[0][:, 0] for M in Ms]
    tvecs = np.array([M[:3, 3] for M in Ms])
    best = get_most_common(np.hstack([rvecs, tvecs]))
    Ms_best = Ms[best]
    return Ms_best


def mean_transform(M_list):
    rvecs = [cv2.Rodrigues(M[:3,:3])[0][:, 0] for M in M_list]
    tvecs = [M[:3, 3] for M in M_list]

    rvec = np.mean(rvecs, axis=0)
    tvec = np.mean(tvecs, axis=0)

    return get_extrinsic_matrix(rvec, tvec)

def mean_transform_robust(M_list, approx=None, error=0.3):
    if approx is None:
        M_list_robust = M_list
    else:
        M_list_robust = []
        for M in M_list:
            rot_error = (M - approx)[:3,:3]
            m = np.max(np.abs(rot_error))
            if m < error:
                M_list_robust.append(M)
    return mean_transform(M_list_robust)


def get_most_common(vals):
    Z = linkage(whiten(vals), 'ward')
    n_clust = max(len(vals)/10, 3)
    clusts = fcluster(Z, t=n_clust, criterion='maxclust')
    cc = Counter(clusts[clusts >= 0])
    most = cc.most_common(n=1)
    top = most[0][0]
    good = clusts == top
    return good


def get_rtvec(M):
    rvec = cv2.Rodrigues(M[:3, :3])[0].flatten()
    tvec = M[:3, 3].flatten()
    return rvec, tvec