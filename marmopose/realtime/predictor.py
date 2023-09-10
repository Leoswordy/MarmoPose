import cv2
import numpy as np
from pathlib import Path
from typing import Dict

import sleap
from marmopose.utils.helpers import get_color_list, Timer
from marmopose.visualization.display_2d import label_image_with_pose

import tensorflow as tf
from sleap.nn.config.training_job import TrainingJobConfig
from sleap.nn.inference import CentroidCrop, TopDownMultiClassFindPeaks, SingleInstanceInferenceLayer
from sleap.nn.utils import reset_input_layer
    

class RealtimeSingleInstancePredictor:
    def __init__(self, config, n_cams: int = 4, crop_size: int = 640, 
                 scale: float = None, peak_threshold: float = 0.2):
        self.config = config
        self.crop_size = crop_size

        self.previous_offsets = np.full((n_cams, 1, 2), np.nan)

        self.build_predictor(config['directory']['model'], scale, peak_threshold)
        self.init_label_params()

    def build_predictor(self, model_dir: str, scale: float = None, peak_threshold: float = 0.2):
        """Build the predictor using the specified model directory, scale, and peak threshold.

        Args:
            model_dir: The directory where the model is stored.
            scale (optional): The scale for processing the input. Defaults to None.
            peak_threshold: The threshold for peak detection. Defaults to 0.2.
        """
        confmap_config = TrainingJobConfig.load_json(model_dir)
        keras_model = tf.keras.models.load_model(Path(model_dir) / "best_model.h5", compile=False)
        keras_model = reset_input_layer(keras_model=keras_model, new_shape=None)

        input_scale = scale if scale is not None else confmap_config.data.preprocessing.input_scaling

        self.predictor = SingleInstanceInferenceLayer(
            keras_model=keras_model,
            input_scale=input_scale,
            pad_to_stride=confmap_config.data.preprocessing.pad_to_stride,
            output_stride=confmap_config.model.heads.single_instance.output_stride,
            peak_threshold=peak_threshold,
            return_confmaps=False
        )
    
    def init_label_params(self) -> None:
        """Initialize label parameters such as skeleton indices and color lists from the configuration."""
        bodyparts = self.config['animal']['bodyparts']
        skeleton = self.config['visualization']['skeleton']
        self.skeleton_indices = [[bodyparts.index(bp) for bp in line] for line in skeleton]

        self.track_color_list = get_color_list(self.config['visualization']['track_cmap'], self.config['animal']['number'])
        self.skeleton_color_list = get_color_list(self.config['visualization']['skeleton_cmap'], len(self.skeleton_indices))

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Predict 2D points and scores from the input images, and concatenates the points with their respective scores.

        Args:
            images: Input images with dimensions (n_cams, height, width, channels).

        Returns:
            Array of 2D points with their respective scores, shape of (n_cams, n_tracks=1, n_bodyparts, 3).
        """
        if np.isnan(self.previous_offsets).any():
            preds = self.predictor(images)

            all_points_2d = preds['instance_peaks'].numpy() # (n_cams, 1, n_bodyparts, 2)
            all_scores_2d = preds['instance_peak_vals'].numpy() # (n_cams, 1, n_bodyparts)
        else:
            crops = self.get_crop_output(images)
            preds = self.predictor(crops)

            all_points_2d = preds['instance_peaks'].numpy()
            all_points_2d = all_points_2d + np.expand_dims(self.previous_offsets, axis=1)
            all_scores_2d = preds['instance_peak_vals'].numpy()

        self.update_offsets(images, all_points_2d)
        self.draw_images(images, all_points_2d)

        all_points_with_score_2d = np.concatenate((all_points_2d, all_scores_2d[..., None]), axis=-1)
        return all_points_with_score_2d
    
    def get_crop_output(self, images: np.ndarray) -> np.ndarray:
        """Generate cropped outputs from the input images based on previous offsets.

        Args:
            images: Input images with dimensions (n_cams, height, width, channels).

        Returns:
            Array of cropped images.
        """
        crops = []
        for image, (x, y) in zip(images, self.previous_offsets.squeeze()):
            x, y = int(x), int(y)
            crops.append(image[y:y+self.crop_size, x:x+self.crop_size])

        crops = np.array(crops)
        return crops
    
    def update_offsets(self, images: np.ndarray, all_points_2d: np.ndarray) -> None:
        """Update the offset values based on the 2D points predicted.

        Args:
            images: Input images shape of (n_cams, height, width, channels).
            all_points_2d: Predicted 2D points shape of (n_cams, n_tracks=1, n_bodyparts, 2)
        """
        _, height, width, _ = images.shape
        n_cams, n_tracks, _, _ = all_points_2d.shape
        for cam_idx in range(n_cams):
            for track_idx in range(n_tracks):
                points = all_points_2d[cam_idx, track_idx]
                if np.isnan(points).all():
                    continue

                min_x, max_x = np.nanmin(points[:, 0]), np.nanmax(points[:, 0])
                min_y, max_y = np.nanmin(points[:, 1]), np.nanmax(points[:, 1])
                min_x = self.adjust_positions(min_x, max_x, width)
                min_y = self.adjust_positions(min_y, max_y, height)

                self.previous_offsets[cam_idx, track_idx] = (min_x, min_y)
    
    def adjust_positions(self, min_pos: float, max_pos: float, max_dim: int) -> int:
        """Adjust positions to avoid going out of bounds when cropping.

        Args:
            min_pos: The minimum position value.
            max_pos: The maximum position value.
            max_dim: The maximum dimension of the image.

        Returns:
            The adjusted minimum position value.
        """
        padding = (self.crop_size - (max_pos - min_pos)) / 2

        if min_pos < padding:
            min_pos = 0
        elif max_dim - max_pos < padding:
            min_pos = max_dim - self.crop_size - 1
        else:
            min_pos -= padding

        return int(min_pos)
    
    def draw_images(self, images: np.ndarray, all_points_2d: np.ndarray) -> None:
        """Draw the predicted points and skeleton structure on the input images.

        Args:
            images: Input images shape of (n_cams, height, width, channels).
            all_points_2d: Predicted 2D points shape of (n_cams, n_tracks=1, n_bodyparts, 2)
        """
        for image, points, offsets in zip(images, all_points_2d, self.previous_offsets):
            label_image_with_pose(image, points, self.skeleton_indices, self.track_color_list, self.skeleton_color_list)
            for x, y in offsets:
                if np.isnan(x) or np.isnan(y):
                    continue
                x, y = int(x), int(y)
                cv2.rectangle(image, (x, y), (x+self.crop_size, y+self.crop_size), (255, 0, 0), 4)


class RealtimeMultiInstacePredictor:
    def __init__(self, config, n_cams: int = 4, crop_size: int = 640, 
                 scale: float = None, peak_threshold: float = 0.2):
        self.config = config
        self.n_cams = n_cams
        self.crop_size = crop_size
        self.n_tracks = config['animal']['number']

        self.previous_centroid = np.full((n_cams, self.n_tracks, 2), np.nan)

        model_dir = config['directory']['model']
        self.build_centroid_predictor(model_dir[0], scale, peak_threshold)
        self.build_centered_instance_predictor(model_dir[1], scale, peak_threshold)

        self.init_label_params()

    def build_centroid_predictor(self, model_dir: str, scale: float = None, peak_threshold: float = 0.2) -> None:
        """Build the centroid predictor with the specified parameters and configuration loaded from the model directory.

        Args:
            model_dir: Directory where the model files are stored.
            scale (optional): Scale factor for the input. Defaults to None.
            peak_threshold (optional): Threshold for peak detection in the model. Defaults to 0.2.
        """
        centroid_config = TrainingJobConfig.load_json(model_dir)
        keras_model = tf.keras.models.load_model(Path(model_dir) / "best_model.h5", compile=False)
        keras_model = reset_input_layer(keras_model=keras_model, new_shape=None)

        input_scale = scale if scale is not None else centroid_config.data.preprocessing.input_scaling

        self.centroid_predictor = CentroidCrop(
            keras_model=keras_model,
            crop_size=self.crop_size,
            input_scale=input_scale,
            pad_to_stride=centroid_config.data.preprocessing.pad_to_stride,
            output_stride=centroid_config.model.heads.centroid.output_stride,
            peak_threshold=peak_threshold,
            return_confmaps=False,
            max_instances=self.n_tracks, 
            return_crops=True
        )

    def build_centered_instance_predictor(self, model_dir: str, scale: float = None, peak_threshold: float = 0.2) -> None:
        """Build the centered instance predictor with the specified parameters and configuration loaded from the model directory.

        Args:
            model_dir: Directory where the model files are stored.
            scale (optional): Scale factor for the input. Defaults to None.
            peak_threshold (optional): Threshold for peak detection in the model. Defaults to 0.2.
        """
        confmap_config = TrainingJobConfig.load_json(model_dir)
        keras_model = tf.keras.models.load_model(Path(model_dir) / "best_model.h5", compile=False)
        keras_model = reset_input_layer(keras_model=keras_model, new_shape=None)

        input_scale = scale if scale is not None else confmap_config.data.preprocessing.input_scaling

        self.centered_instance_predictor = TopDownMultiClassFindPeaks(
            keras_model=keras_model,
            input_scale=input_scale,
            output_stride=confmap_config.model.heads.multi_class_topdown.confmaps.output_stride,
            peak_threshold=peak_threshold,
            return_confmaps=False
        )
    
    def init_label_params(self) -> None:
        """Initialize label parameters such as skeleton indices and color lists from the configuration."""
        bodyparts = self.config['animal']['bodyparts']
        skeleton = self.config['visualization']['skeleton']
        self.skeleton_indices = [[bodyparts.index(bp) for bp in line] for line in skeleton]

        self.track_color_list = get_color_list(self.config['visualization']['track_cmap'], self.config['animal']['number'])
        self.skeleton_color_list = get_color_list(self.config['visualization']['skeleton_cmap'], len(self.skeleton_indices))
    
    def predict(self, images: np.ndarray) -> np.ndarray:
        """Predict 2D points and scores from the input images, and concatenates the points with their respective scores.

        Args:
            images: Input images with dimensions (n_cams, height, width, channels).

        Returns:
            Array of 2D points with their respective scores, shape of (n_cams, n_tracks=1, n_bodyparts, 3).
        """
        crop_output = self.get_crop_output(images)

        peaks_output = self.centered_instance_predictor(crop_output)

        all_points_2d = peaks_output['instance_peaks'].numpy() # (n_cams, n_tracks, n_bodyparts, 2)
        all_scores_2d = peaks_output['instance_peak_vals'].numpy() # (n_cams, n_tracks, n_bodyparts)
        self.update_centroids(all_points_2d)
        self.draw_images(images, all_points_2d)

        all_points_with_score_2d = np.concatenate((all_points_2d, all_scores_2d[..., None]), axis=-1)
        return all_points_with_score_2d
    
    def get_crop_output(self, images: np.ndarray) -> Dict:
        """Get the crop output either by predicting new crops or by using previously calculated centroids.

        Args:
            images: Input images shape of (n_cams, height, width, channels).

        Returns:
            Dictionary containing the crop outputs.
        """
        # timer = Timer().start()
        # if np.isnan(self.previous_centroid).any():
        #     crop_output = self.centroid_predictor(images)
        #     timer.record('predict crops')
        # else:
        #     crop_output = self.get_crop_output_from_previous_centorid(images)
        #     timer.record('previous crops')
        # timer.show()
        crop_output = self.centroid_predictor(images)
        return crop_output

    @tf.function
    def get_crop_output_from_previous_centorid(self, images: np.ndarray) -> Dict:
        """Compute the crop output based on previously calculated centroids.

        Args:
            images: Input images shape of (n_cams, height, width, channels).

        Returns:
            Dictionary containing the crop outputs based on previous centroids.
        """
        centroid_points = tf.reshape(tf.constant(self.previous_centroid, dtype=tf.float32), [-1, 2])
        
        arange_values = tf.range(self.n_cams, dtype=tf.int32)
        crop_sample_inds = tf.repeat(arange_values, self.n_tracks)
        
        bboxes = sleap.nn.data.instance_cropping.make_centered_bboxes(centroid_points, self.crop_size, self.crop_size)
        crops = sleap.nn.peak_finding.crop_bboxes(images, bboxes, crop_sample_inds)

        crop_output = {
            'crops': crops,
            'crop_offsets': tf.RaggedTensor.from_tensor(self.previous_centroid-self.crop_size/2),
            'samples': self.n_cams,
            'crop_sample_inds': crop_sample_inds
        }
        return crop_output

    def update_centroids(self, all_points_2d: np.ndarray) -> None:
        """Update the centroids based on the newly predicted 2D points.

        Args:
            all_points_2d: Array containing the newly predicted 2D points, shape of (n_cams, n_tracks, n_bodyparts, 2).
        """
        n_cams, n_tracks, _, _ = all_points_2d.shape
        for cam_idx in range(n_cams):
            for track_idx in range(n_tracks):
                points = all_points_2d[cam_idx, track_idx]
                if not np.isnan(points[8]).any():
                    self.previous_centroid[cam_idx, track_idx] = points[8]
                elif not np.isnan(points).all():
                    self.previous_centroid[cam_idx, track_idx] = np.nanmean(points, axis=0)
                else:
                    self.previous_centroid[cam_idx, track_idx] = (np.nan, np.nan)
    
    def draw_images(self, images: np.ndarray, all_points_2d: np.ndarray) -> None:
        """Draw the predicted points and bounding boxes on the images.

        Args:
            images: Input images shape of (n_cams, height, width, channels).
            all_points_2d: Predicted 2D points shape of (n_cams, n_tracks, n_bodyparts, 2)
        """
        for image, points, offsets in zip(images, all_points_2d, self.previous_centroid):
            label_image_with_pose(image, points, self.skeleton_indices, self.track_color_list, self.skeleton_color_list)
            for x, y in offsets:
                if np.isnan(x) or np.isnan(y):
                    continue
                cv2.rectangle(image, (int(x-self.crop_size/2), int(y-self.crop_size/2)), (int(x+self.crop_size/2), int(y+self.crop_size/2)), (255, 0, 0), 4)