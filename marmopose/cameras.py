import cv2
import toml
import numpy as np
from numba import jit
from tqdm import trange
from scipy import signal, optimize
from scipy.sparse import dok_matrix

from typing import List, Tuple, Dict, Optional, Any


@jit(nopython=True, parallel=True)
def triangulate_simple(points: List[Tuple[float, float]], camera_mats: List[np.ndarray]) -> np.ndarray:
    """
    Computes the 3D position of a point from multiple camera views using simple triangulation.

    Args:
        points: List of 2D points from each camera view.
        camera_mats: List of camera projection matrices.

    Returns:
        3D position of the point.
    """
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


def make_M(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
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


def interpolate_data(values: np.ndarray) -> np.ndarray:
    """
    Interpolates data to fill NaN values.

    Args:
        values: The data to be interpolated.

    Returns:
        The interpolated data.
    """
    nans = np.isnan(values)
    idx = lambda z: np.nonzero(z)[0]
    out = np.copy(values)
    out[nans] = np.interp(idx(nans), idx(~nans), values[~nans]) if not np.isnan(values).all() else 0
    return out


def medfilt_data(values: np.ndarray, size: int = 15) -> np.ndarray:
    """
    Applies a median filter to the data.

    Args:
        values: The data to be filtered.
        size: The size of the median filter. Defaults to 15.

    Returns:
        The filtered data.
    """
    padsize = size + 5
    vpad = np.pad(values, (padsize, padsize), mode='reflect')
    vpadf = signal.medfilt(vpad, kernel_size=size)
    return vpadf[padsize:-padsize]


class Camera:
    def __init__(self,
                 matrix: np.ndarray = np.eye(3),
                 dist: np.ndarray = np.zeros(5),
                 size: Optional[Tuple[int, int]] = None,
                 rvec: np.ndarray = np.zeros(3),
                 tvec: np.ndarray = np.zeros(3),
                 name: Optional[str] = None,
                 extra_dist: bool = False):
        """
        Initialize a Camera object.

        Args:
            matrix: Camera matrix. Defaults to np.eye(3).
            dist: Distortions array. Defaults to np.zeros(5).
            size: Size of the image. Defaults to None.
            rvec: Rotation vector. Defaults to np.zeros(3).
            tvec: Translation vector. Defaults to np.zeros(3).
            name: Name of the camera. Defaults to None.
            extra_dist: Boolean flag for extra distortion. Defaults to False.
        """
        self.set_camera_matrix(matrix)
        self.set_distortions(dist)
        self.set_size(size)
        self.set_rotation(rvec)
        self.set_translation(tvec)
        self.set_name(name)
        self.extra_dist = extra_dist
    
    def load_dict(self, d: dict) -> None:
        """
        Load camera parameters from a dictionary.

        Args:
            d: A dictionary containing camera parameters.
        """
        self.set_camera_matrix(d['matrix'])
        self.set_rotation(d['rotation'])
        self.set_translation(d['translation'])
        self.set_distortions(d['distortions'])
        self.set_name(d['name'])
        self.set_size(d['size'])
    
    @classmethod
    def from_dict(cls, d: dict) -> 'Camera':
        """
        Create a Camera object from a dictionary.

        Args:
            d: A dictionary containing camera parameters.

        Returns:
            A Camera object.
        """
        cam = cls()
        cam.load_dict(d)
        return cam
    
    def get_dict(self) -> Dict:
        """
        Get a dictionary representing the camera.

        Returns:
            A dictionary representing the Camera object.
        """
        return {
            'name': self.get_name(),
            'size': list(self.get_size()),
            'matrix': self.get_camera_matrix().tolist(),
            'distortions': self.get_distortions().tolist(),
            'rotation': self.get_rotation().tolist(),
            'translation': self.get_translation().tolist(),
        }
    
    def set_camera_matrix(self, matrix):
        """Set the camera matrix."""
        self.matrix = np.array(matrix, dtype='float64')
    
    def get_camera_matrix(self):
        """Get the camera matrix."""
        return self.matrix

    def set_distortions(self, dist):
        """Set the distortions."""
        self.dist = np.array(dist, dtype='float64').ravel()
    
    def get_distortions(self):
        """Get the distortions."""
        return self.dist

    def set_rotation(self, rvec):
        """Set the rotation vector."""
        self.rvec = np.array(rvec, dtype='float64').ravel()
    
    def get_rotation(self):
        """Get the rotation vector."""
        return self.rvec
    
    def set_translation(self, tvec):
        """Set the translation vector."""
        self.tvec = np.array(tvec, dtype='float64').ravel()
    
    def get_translation(self):
        """Get the translation vector."""
        return self.tvec
    
    def set_name(self, name):
        """Set the name of the camera."""
        self.name = str(name)
    
    def get_name(self):
        """Get the name of the camera."""
        return self.name
    
    def set_size(self, size: Tuple[int, int]) -> None:
        """
        Set the size of the camera.

        Args:
            size: A tuple representing the size (width, height).
        """
        self.size = size
    
    def get_size(self) -> Tuple[int, int]:
        """
        Get the size of the camera.

        Returns:
            A tuple representing the size (width, height).
        """
        return self.size
    
    def resize_camera(self, scale: float) -> None:
        """Resize the camera by scale factor, updating intrinsics to match
        
        Args:
            scale: The size scale factor
        """
        size = self.get_size()
        new_size = size[0] * scale, size[1] * scale
        matrix = self.get_camera_matrix()
        new_matrix = matrix * scale
        new_matrix[2, 2] = 1
        self.set_size(new_size)
        self.set_camera_matrix(new_matrix)
    
    def get_extrinsics_mat(self) -> np.ndarray:
        """
        Get the extrinsics matrix.

        Returns:
            The extrinsics matrix.
        """
        return make_M(self.rvec, self.tvec)
    
    def distort_points(self, points: np.ndarray) -> np.ndarray:
        """
        Apply distortions to points using the camera's intrinsic parameters.

        Args:
            points: 2D array of points (x, y) to be distorted.

        Returns:
            2D array of distorted points.
        """
        reshaped_points = points.reshape(-1, 1, 2)
        homogeneous_points = np.dstack([reshaped_points, np.ones(points.shape[0], 1, 1)])
        distorted_points, _ = cv2.projectPoints(homogeneous_points, np.zeros(3), np.zeros(3), self.matrix, self.dist)
        return distorted_points.reshape(points.shape)
    
    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """
        Remove distortions from points using the camera's intrinsic parameters.

        Args:
            points: 2D array of distorted points (x, y).

        Returns:
            2D array of undistorted points.
        """
        reshaped_points = points.reshape(-1, 1, 2)
        undistorted_points = cv2.undistortPoints(reshaped_points, self.matrix, self.dist)
        return undistorted_points.reshape(points.shape)

    def project(self, points: np.ndarray) -> np.ndarray:
        """
        Project 3D points to 2D image plane using camera's extrinsic and intrinsic parameters.

        Args:
            points: 3D array of points (x, y, z) to be projected.

        Returns:
            2D array of projected points.
        """
        reshaped_points = points.reshape(-1, 1, 3)
        projected_points, _ = cv2.projectPoints(reshaped_points, self.rvec, self.tvec, self.matrix, self.dist)
        return projected_points

    def reprojection_error(self, p3d: np.ndarray, p2d: np.ndarray) -> np.ndarray:
        """
        Compute the reprojection error for 3D points against their 2D projections.

        Args:
            p3d: 3D array of points.
            p2d: 2D array of corresponding projected points.

        Returns:
            Array of differences between original and reprojected 2D points.
        """
        projected_points = self.project(p3d).reshape(p2d.shape)
        return np.abs(p2d - projected_points)
    
    def copy(self) -> 'Camera':
        """
        Create a copy of the current Camera object.

        Returns:
            A copy of the current Camera object.
        """
        return Camera(matrix = self.get_camera_matrix().copy(),
                      dist = self.get_distortions().copy(),
                      size = self.get_size(),
                      rvec = self.get_rotation().copy(),
                      tvec = self.get_translation().copy(),
                      name = self.get_name(),
                      extra_dist = self.extra_dist)


class FisheyeCamera(Camera):
    def __init__(self,
                 matrix: np.ndarray = np.eye(3),
                 dist: np.ndarray = np.zeros(4),
                 size: Optional[Tuple[int, int]] = None,
                 rvec: np.ndarray = np.zeros(3),
                 tvec: np.ndarray = np.zeros(3),
                 name: Optional[str] = None,
                 extra_dist: bool = False):
        """
        Initialize a FisheyeCamera instance.

        Args:
            matrix: The camera matrix.
            dist: The distortion coefficients.
            size: The size (width, height) of the image.
            rvec: The rotation vector.
            tvec: The translation vector.
            name: The name of the camera.
            extra_dist: Indicator of extra distortion.
        """
        super().__init__(matrix, dist, size, rvec, tvec, name, extra_dist)

    def get_dict(self) -> Dict[str, Any]:
        """
        Get the dictionary representation of the camera.

        Returns:
            The dictionary representation of the camera.
        """
        d = super().get_dict()
        d['fisheye'] = True
        return d

    def distort_points(self, points: np.ndarray) -> np.ndarray:
        """
        Applies distortion to the given points.

        Args:
            points: The points to distort.

        Returns:
            The distorted points.
        """
        reshaped_points = points.reshape(-1, 1, 2)
        homogeneous_points = np.dstack([reshaped_points, np.ones((reshaped_points.shape[0], 1, 1))])
        distorted_points, _ = cv2.fisheye.projectPoints(homogeneous_points, np.zeros(3), np.zeros(3), self.matrix, self.dist)
        return distorted_points.reshape(points.shape)

    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """
        Removes distortion from the given points.

        Args:
            points: The points to undistort.

        Returns:
            The undistorted points.
        """
        reshaped_points = points.reshape(-1, 1, 2)
        undistorted_points = cv2.fisheye.undistortPoints(reshaped_points.astype('float64'), self.matrix, self.dist)
        return undistorted_points.reshape(points.shape)

    def project(self, points: np.ndarray) -> np.ndarray:
        """
        Projects 3D points to 2D.

        Args:
            points: The 3D points to project.

        Returns:
            The 2D projections.
        """
        reshaped_points = points.reshape(-1, 1, 3)
        projected_points, _ = cv2.fisheye.projectPoints(reshaped_points, self.rvec, self.tvec, self.matrix, self.dist)
        return projected_points
    
    def copy(self) -> 'FisheyeCamera':
        """
        Create a copy of the current FisheyeCamera object.

        Returns:
            A copy of the current FisheyeCamera object.
        """
        return FisheyeCamera(matrix = self.get_camera_matrix().copy(), 
                             dist = self.get_distortions().copy(), 
                             size = self.get_size(), 
                             rvec = self.get_rotation().copy(), 
                             tvec = self.get_translation().copy(), 
                             name = self.get_name(), 
                             extra_dist = self.extra_dist)


class CameraGroup:
    def __init__(self, cameras: List, metadata: Dict = {}):
        """Initializes a CameraGroup object with given cameras and metadata.

        Args:
            cameras: List of Camera or FisheyeCamera objects.
            metadata: A dictionary of metadata. Defaults to None.
        """
        self.cameras = cameras
        self.metadata = metadata
    
    @classmethod
    def from_dicts(cls, dicts: List[Dict]) -> 'CameraGroup':
        """Class method that creates a CameraGroup object from a list of dictionaries.

        Args:
            dicts: A list of dictionaries representing camera attributes.

        Returns:
            CameraGroup object.
        """
        cameras = [FisheyeCamera.from_dict(d) if d.get('fisheye') else Camera.from_dict(d) for d in dicts]
        return cls(cameras)
    
    @staticmethod
    def load(path: str) -> 'CameraGroup':
        """Loads CameraGroup from a TOML file.

        Args:
            path: Path of the TOML file.

        Returns:
            CameraGroup object.
        """
        master_dict = toml.load(path)
        items = [v for k, v in sorted(master_dict.items()) if k != 'metadata']
        cgroup = CameraGroup.from_dicts(items)
        cgroup.metadata = master_dict.get('metadata', {})
        return cgroup

    def get_names(self) -> List[str]:
        """Returns the names of the cameras.

        Returns:
            List of camera names.
        """
        return [cam.get_name() for cam in self.cameras]
    
    def resize_cameras(self, scale: float) -> None:
        """Resize the cameras in the group.

        Args:
            scale: The resize scale.
        """
        for cam in self.cameras:
            cam.resize_camera(scale)

    def subset_cameras(self, indices: List[int]) -> 'CameraGroup':
        """Subsets the cameras based on given indices and returns a new CameraGroup.

        Args:
            indices: A list of indices.

        Returns:
            New CameraGroup object with subset of cameras.
        """
        return CameraGroup([self.cameras[ix].copy() for ix in indices], self.metadata)

    def subset_cameras_names(self, names: List[str]) -> 'CameraGroup':
        """Subsets the cameras based on given names and returns a new CameraGroup.

        Args:
            names: A list of camera names.

        Returns:
            New CameraGroup object with subset of cameras.

        Raises:
            IndexError: If a name is not part of camera names.
        """
        cur_names_dict = {name: idx for idx, name in enumerate(self.get_names())}
        indices = [cur_names_dict[name] for name in names if name in cur_names_dict]
        if len(names) != len(indices):
            missing_names = set(names) - set(cur_names_dict.keys())
            raise IndexError(f"names {missing_names} not part of camera names: {list(cur_names_dict.keys())}")
        return self.subset_cameras(indices)
    
    def triangulate(self, points: np.ndarray, undistort: bool = True, show_progress: bool = False) -> np.ndarray:
        """
        Given a CxNx2 array of points, triangulates the points to get their 3D coordinates.

        Args:
            points: A CxNx2 array, where N is the number of points, C is the number of cameras and 2 is the 2D coordinates.
            undistort: If True, undistorts the points using the camera parameters. Default is True.
            show_progress: If True, displays a progress bar. Default is False.

        Returns:
            An Nx3 array of points containing the triangulated 3D coordinates of the points.
        """

        if undistort:
            points = np.array([cam.undistort_points(np.copy(pt)) for pt, cam in zip(points, self.cameras)])

        cam_mats = np.array([cam.get_extrinsics_mat() for cam in self.cameras])

        n_points = points.shape[1]
        out = np.full((n_points, 3), np.nan)

        iterator = trange(n_points, ncols=100, desc='Triangulating... ', unit='points') if show_progress else range(n_points)

        for ip in iterator:
            sub_points = points[:, ip, :]
            valid_points = ~np.isnan(sub_points[:, 0])
            if np.sum(valid_points) >= 2:
                out[ip] = triangulate_simple(sub_points[valid_points], cam_mats[valid_points])

        return out

    def _initialize_params_triangulation(self, 
                                         p3ds: np.ndarray, 
                                         constraints: List[Tuple[int, int]] = [], 
                                         constraints_weak: List[Tuple[int, int]] = []) -> np.ndarray:
        """
        Initialize the parameters for the least squares optimization problem in 3D triangulation.
        
        Args:
            p3ds: Array containing the 3D coordinates of points.
            constraints: List of pairs of indices, each representing a rigid link between two points. (Optional)
            constraints_weak: List of pairs of indices, each representing a weakly rigid link between two points. (Optional)
            
        Returns:
            The initial guess for the parameters of the optimization problem, obtained 
            by concatenating the flattened `p3ds` array with the lengths of the links.
        """

        # Predefined measured lengths of rigid links and weakly rigid links
        joint_lengths = np.array([30, 30, 40, 40, 80, 80, 140, 140, 70, 70, 70, 70], dtype='float64')
        joint_lengths_weak = np.array([70, 70, 80, 80], dtype='float64')

        # Concatenate the flattened p3ds array with the joint lengths to form the initial guess
        return np.hstack([p3ds.ravel(), joint_lengths, joint_lengths_weak])
    
    def _jac_sparsity_triangulation(self, 
                                    p2ds: np.ndarray, 
                                    constraints: List[Tuple[int, int]] = [], 
                                    constraints_weak: List[Tuple[int, int]] = [], 
                                    n_deriv_smooth: int = 1) -> dok_matrix:
        """
        Create the sparsity pattern of the Jacobian for the least squares optimization problem in 3D triangulation.
        
        Args:
            p2ds: Array containing the 2D projections of the 3D points.
            constraints: List of pairs of indices, each representing a rigid link between two points. (Optional)
            constraints_weak: List of pairs of indices, each representing a weakly rigid link between two points. (Optional)
            n_deriv_smooth: Order of the derivative used for the smoothness constraint. (Optional)
            
        Returns:
            A sparse matrix that describes the sparsity pattern of the Jacobian matrix.
        """

        n_cams, n_frames, n_joints, _ = p2ds.shape
        n_constraints = len(constraints)
        n_constraints_weak = len(constraints_weak)

        p2ds_flat = p2ds.reshape((n_cams, -1, 2))

        point_indices = np.zeros(p2ds_flat.shape, dtype='int32')
        for i in range(p2ds_flat.shape[1]):
            point_indices[:, i] = i

        point_indices_3d = np.arange(n_frames*n_joints)\
                             .reshape((n_frames, n_joints))

        valid_points = ~np.isnan(p2ds_flat)
        n_errors_reproj = np.sum(valid_points)
        n_errors_smooth = (n_frames-n_deriv_smooth) * n_joints * 3
        n_errors_lengths = n_constraints * n_frames
        n_errors_lengths_weak = n_constraints_weak * n_frames

        n_errors = n_errors_reproj + n_errors_smooth + \
            n_errors_lengths + n_errors_lengths_weak

        n_3d = n_frames*n_joints*3
        n_params = n_3d + n_constraints + n_constraints_weak

        point_indices_good = point_indices[valid_points]

        A_sparse = dok_matrix((n_errors, n_params), dtype='int16')

        # constraints for reprojection errors
        ix_reproj = np.arange(n_errors_reproj)
        for k in range(3):
            A_sparse[ix_reproj, point_indices_good * 3 + k] = 1

        # sparse constraints for smoothness in time
        frames = np.arange(n_frames-n_deriv_smooth)
        for j in range(n_joints):
            for n in range(n_deriv_smooth+1):
                pa = point_indices_3d[frames, j]
                pb = point_indices_3d[frames+n, j]
                for k in range(3):
                    A_sparse[n_errors_reproj + pa*3 + k, pb*3 + k] = 1

        ## -- strong constraints --
        # joint lengths should change with joint lengths errors
        start = n_errors_reproj + n_errors_smooth
        frames = np.arange(n_frames)
        for cix, (a, b) in enumerate(constraints):
            A_sparse[start + cix*n_frames + frames, n_3d+cix] = 1

        # points should change accordingly to match joint lengths too
        frames = np.arange(n_frames)
        for cix, (a, b) in enumerate(constraints):
            pa = point_indices_3d[frames, a]
            pb = point_indices_3d[frames, b]
            for k in range(3):
                A_sparse[start + cix*n_frames + frames, pa*3 + k] = 1
                A_sparse[start + cix*n_frames + frames, pb*3 + k] = 1

        ## -- weak constraints --
        # joint lengths should change with joint lengths errors
        start = n_errors_reproj + n_errors_smooth + n_errors_lengths
        frames = np.arange(n_frames)
        for cix, (a, b) in enumerate(constraints_weak):
            A_sparse[start + cix*n_frames + frames, n_3d + n_constraints + cix] = 1

        # points should change accordingly to match joint lengths too
        frames = np.arange(n_frames)
        for cix, (a, b) in enumerate(constraints_weak):
            pa = point_indices_3d[frames, a]
            pb = point_indices_3d[frames, b]
            for k in range(3):
                A_sparse[start + cix*n_frames + frames, pa*3 + k] = 1
                A_sparse[start + cix*n_frames + frames, pb*3 + k] = 1

        return A_sparse

    @jit(forceobj=True, parallel=True)
    def _error_fun_triangulation(self, 
                                params: np.ndarray, 
                                p2ds: np.ndarray,
                                constraints: List[Tuple[int, int]] = [], 
                                constraints_weak: List[Tuple[int, int]] = [], 
                                scores: np.ndarray = None,
                                scale_smooth: float = 4,
                                scale_length: float = 2,
                                scale_length_weak: float = 0.5,
                                reproj_error_threshold: float = 15,
                                reproj_loss: str = 'soft_l1',
                                n_deriv_smooth: int = 1) -> np.ndarray:
        """
        Compute the error vector for the least squares problem in 3D triangulation.

        Args:
            params: Parameters to be optimized.
            p2ds: Observed 2D projections of the 3D points.
            constraints: List of pairs of indices, each representing a rigid link between two points. (Optional)
            constraints_weak: List of pairs of indices, each representing a weakly rigid link between two points. (Optional)
            scores: Scores for the points. (Optional)
            scale_smooth: Scale for the smoothness term in the error function. (Optional)
            scale_length: Scale for the length term in the error function for strong constraints. (Optional)
            scale_length_weak: Scale for the length term in the error function for weak constraints. (Optional)
            reproj_error_threshold: Threshold for the reprojection error. (Optional)
            reproj_loss: Type of loss function to use for the reprojection error. (Optional)
            n_deriv_smooth: Order of the derivative used for the smoothness constraint. (Optional)

        Returns:
            The error vector for the least squares problem.
        """

        n_cams, n_frames, n_joints, _ = p2ds.shape

        n_3d = n_frames*n_joints*3
        n_constraints = len(constraints)
        n_constraints_weak = len(constraints_weak)

        # Load parameters
        p3ds = params[:n_3d].reshape((n_frames, n_joints, 3))
        joint_lengths = np.array(params[n_3d:n_3d+n_constraints])
        joint_lengths_weak = np.array(params[n_3d+n_constraints:])

        # Compute reprojection errors
        p3ds_flat = p3ds.reshape(-1, 3)
        p2ds_flat = p2ds.reshape((n_cams, -1, 2))
        errors = self.reprojection_error(p3ds_flat, p2ds_flat)
        if scores is not None:
            scores_flat = scores.reshape((n_cams, -1))
            errors = errors * scores_flat[:, :, None]
        errors_reproj = errors[~np.isnan(p2ds_flat)]

        rp = reproj_error_threshold
        errors_reproj = np.abs(errors_reproj)
        if reproj_loss == 'huber':
            bad = errors_reproj > rp
            errors_reproj[bad] = rp*(2*np.sqrt(errors_reproj[bad]/rp) - 1)
        elif reproj_loss == 'linear':
            pass
        elif reproj_loss == 'soft_l1':
            errors_reproj = rp*2*(np.sqrt(1+errors_reproj/rp)-1)

        # Temporal constraint
        errors_smooth = np.diff(p3ds, n=n_deriv_smooth, axis=0).ravel() * scale_smooth

        # Joint length constraint
        errors_lengths = np.empty((n_constraints, n_frames), dtype='float64')
        for cix, (a, b) in enumerate(constraints):
            lengths = np.linalg.norm(p3ds[:, a] - p3ds[:, b], axis=1)
            expected = joint_lengths[cix]
            errors_lengths[cix] = 100*(lengths - expected)/expected
        errors_lengths = errors_lengths.ravel() * scale_length

        errors_lengths_weak = np.empty((n_constraints_weak, n_frames), dtype='float64')
        for cix, (a, b) in enumerate(constraints_weak):
            lengths = np.linalg.norm(p3ds[:, a] - p3ds[:, b], axis=1)
            expected = joint_lengths_weak[cix]
            errors_lengths_weak[cix] = 100*(lengths - expected)/expected
        errors_lengths_weak = errors_lengths_weak.ravel() * scale_length_weak

        return np.hstack([errors_reproj, errors_smooth, errors_lengths, errors_lengths_weak])

    def optim_points(self, 
                     points: np.ndarray, 
                     p3ds: np.ndarray,
                     scores: np.ndarray = None, 
                     constraints: List[Tuple[int, int]] = [], 
                     constraints_weak: List[Tuple[int, int]] = [],
                     scale_smooth: float = 4,
                     scale_length: float = 2, 
                     scale_length_weak: float = 0.5,
                     reproj_error_threshold: float = 15, 
                     reproj_loss: str = 'soft_l1',
                     n_deriv_smooth: int = 1, 
                     verbose: bool = False) -> np.ndarray:
        """
        Take in an array of 2D points of shape CxNxJx2, an array of 3D points of shape NxJx3,
        and an array of constraints of shape Kx2, where
            C: number of cameras
            N: number of frames
            J: number of joints
            K: number of constraints

        This function creates an optimized array of 3D points of shape NxJx3.

        Args:
            points: 2D points with shape (C, N, J, 2).
            p3ds: 3D points with shape (N, J, 3).
            scores: Scores for the points.
            constraints: List of pairs of indices, each representing a rigid link between two points.
            constraints_weak: List of pairs of indices, each representing a weakly rigid link between two points.
            scale_smooth: Scale for the smoothness term in the error function.
            scale_length: Scale for the length term in the error function for strong constraints.
            scale_length_weak: Scale for the length term in the error function for weak constraints.
            reproj_error_threshold: Threshold for the reprojection error.
            reproj_loss: Type of loss function to use for the reprojection error.
            n_deriv_smooth: Order of the derivative used for the smoothness constraint.
            verbose: Flag for verbose output.

        Returns:
            Optimized 3D points with shape (N, J, 3).

        Example constraints:
            constraints = [[0, 1], [1, 2], [2, 3]]
            (meaning that lengths of segments 0->1, 1->2, 2->3 are all constant)
        """

        constraints = np.array(constraints)
        constraints_weak = np.array(constraints_weak)

        p3ds_intp = np.apply_along_axis(interpolate_data, 0, p3ds)
        p3ds_med = np.apply_along_axis(medfilt_data, 0, p3ds_intp, size=7)

        default_smooth = 1.0/np.mean(np.abs(np.diff(p3ds_med, axis=0)))
        scale_smooth_full = scale_smooth * default_smooth

        x0 = self._initialize_params_triangulation(p3ds_intp, constraints, constraints_weak)
        x0[~np.isfinite(x0)] = 0

        jac = self._jac_sparsity_triangulation(points, constraints, constraints_weak, n_deriv_smooth)

        opt2 = optimize.least_squares(self._error_fun_triangulation,
                                      x0=x0, 
                                      jac_sparsity=jac,
                                      loss='linear',
                                      ftol=1e-3,
                                      verbose=2*verbose,
                                      args=(points,
                                            constraints,
                                            constraints_weak,
                                            scores,
                                            scale_smooth_full,
                                            scale_length,
                                            scale_length_weak,
                                            reproj_error_threshold,
                                            reproj_loss,
                                            n_deriv_smooth))

        p3ds_optimized = opt2.x[:p3ds.size].reshape(p3ds.shape)

        return p3ds_optimized

    @jit(parallel=True, forceobj=True)
    def reprojection_error(self, 
                           p3ds: np.ndarray, 
                           p2ds: np.ndarray, 
                           mean: bool = False) -> np.ndarray:
        """
        Given an Nx3 array of 3D points and an CxNx2 array of 2D points,
        where N is the number of points and C is the number of cameras,
        this returns an CxNx2 array of errors.

        Args:
            p3ds: 3D points with shape (N, 3).
            p2ds: 2D points with shape (C, N, 2).
            mean: Whether to average the errors and return array of length N of errors.

        Returns:
            Errors with shape (C, N, 2).
        """

        n_cams, n_points, _ = p2ds.shape

        errors = np.empty((n_cams, n_points, 2))

        for cnum, cam in enumerate(self.cameras):
            errors[cnum] = cam.reprojection_error(p3ds, p2ds[cnum])

        if mean:
            errors_norm = np.linalg.norm(errors, axis=2)
            good = ~np.isnan(errors_norm)
            errors_norm[~good] = 0
            denom = np.sum(good, axis=0).astype('float64')
            denom[denom < 1.5] = np.nan
            errors = np.sum(errors_norm, axis=0) / denom

        return errors