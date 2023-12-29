import time
import itertools
import numpy as np
import time

import tensorflow as tf
from keras import backend as K

from marmopose.utils.helpers import orthogonalize_vector


class VariationalAutoencoder:
    def __init__(self, input_dim, hidden_dim=64, latent_dim=20, batch_size=4096,
                 bodyparts=None, skeleton_constraints=None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.autoencoder = self.build_autoencoder()

        self.poseprocessor = PoseProcessor(bodyparts=bodyparts, original_point='spinemid', x_point='neck', xz_point='tailbase', scale_length=6)
        self.skeleton_constraints = skeleton_constraints

    def build_encoder(self):
        inputs = tf.keras.layers.Input(shape=self.input_dim, name='encoder_input')
        x = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(inputs)
        x = tf.keras.layers.Dense(self.hidden_dim*2, activation='relu')(x)
        z_mean = tf.keras.layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = tf.keras.layers.Dense(self.latent_dim, name='z_log_var')(x)
        z = tf.keras.layers.Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])
        
        return tf.keras.models.Model(inputs, [z_mean, z_log_var, z], name='encoder')

    def build_decoder(self):
        latent_inputs = tf.keras.layers.Input(shape=(self.latent_dim,), name='z_sampling')
        x = tf.keras.layers.Dense(self.hidden_dim*2, activation='relu')(latent_inputs)
        x = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.input_dim[0], activation='linear')(x)

        return tf.keras.models.Model(latent_inputs, outputs, name='decoder')

    def build_autoencoder(self):
        inputs = tf.keras.layers.Input(shape=self.input_dim, name='autoencoder_input')
        z_mean, z_log_var, z = self.encoder(inputs)
        outputs = self.decoder(z)
        autoencoder = tf.keras.models.Model(inputs, [outputs, z_mean, z_log_var], name='vae')

        return autoencoder

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def compute_kl_loss(self, z_mean, z_log_var):
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return kl_loss
    
    def compute_limb_loss(self, y_pred):
        y_pred = tf.reshape(y_pred, (-1, 16, 3))

        errors = []
        for (bp1, bp2), expected_length in self.skeleton_constraints:
            actual_lengths = tf.norm(y_pred[:, bp1] - y_pred[:, bp2], axis=1)
            errors.append(tf.abs(actual_lengths - expected_length/10))

        errors_stacked = tf.stack(errors, axis=0)
        return tf.reduce_sum(errors_stacked, axis=0)
    
    def preprocess(self, X, y):
        """
        Args:
            X: (n_samples, n_bodyparts, 3) with masked data (np.nan)
            y: (n_samples, n_bodyparts, 3)

        Returns:
            Preprocessed X and y
        """
        assert X.shape == y.shape 
        n_samples = X.shape[0]

        X = self.poseprocessor.replace_nan(X)
        X = X.reshape(n_samples, -1)

        y = y.reshape(n_samples, -1)
        return X, y

    def train(self, X_train, y_train, X_val, y_val, epochs=100, lr=0.001, patience=10):
        self.autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

        X_train, y_train = self.preprocess(X_train, y_train)
        X_val, y_val = self.preprocess(X_val, y_val)

        best_val_loss = float('inf')
        no_improve_epochs = 0

        for epoch in range(epochs):
            start_time = time.time()
            p = np.random.permutation(len(X_train))
            X_train_shuffled, y_train_shuffled = X_train[p], y_train[p]

            training_losses = self.process_batch(X_train_shuffled, y_train_shuffled, training=True)
            val_losses = self.evaluate(X_val, y_val)

            print(f'Epoch {epoch + 1}/{epochs} | Time: {time.time() - start_time:.2f}s')
            print(f'Train - Loss: {training_losses[0]:.3f} - MSE Loss: {training_losses[1]:.3f} - Limb Loss: {training_losses[2]:.3f} - KL Loss: {training_losses[3]:.3f}')
            print(f'Val - Loss: {val_losses[0]:.3f} - MSE Loss: {val_losses[1]:.3f} - Limb Loss: {val_losses[2]:.3f} - KL Loss: {val_losses[3]:.3f}')

            if val_losses[0] < best_val_loss:
                best_val_loss = val_losses[0]
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    print(f'Early stopping. Loss does not decrease for {patience} epochs.')
                    break

    def process_batch(self, X, y, training=True):
        total_loss, total_mse_loss, total_limb_loss, total_kl_loss = 0, 0, 0, 0
        num_batches = 0
        for i in range(0, len(X), self.batch_size):
            X_batch = X[i:i + self.batch_size]
            y_batch = y[i:i + self.batch_size]
            num_batches += 1

            with tf.GradientTape() as tape:
                y_pred, z_mean, z_log_var = self.autoencoder(X_batch, training=training)
                kl_loss = self.compute_kl_loss(z_mean, z_log_var)
                mse_loss = tf.keras.losses.mean_squared_error(y_batch, y_pred)
                loss = K.mean(mse_loss + kl_loss)
                # limb_loss = self.compute_limb_loss(y_pred)
                # loss = K.mean(mse_loss + kl_loss + limb_loss*1)

            if training:
                gradients = tape.gradient(loss, self.autoencoder.trainable_variables)
                self.autoencoder.optimizer.apply_gradients(zip(gradients, self.autoencoder.trainable_variables))

            total_loss += loss.numpy()
            total_mse_loss += np.mean(mse_loss)
            total_kl_loss += np.mean(kl_loss)
            # total_limb_loss += np.mean(limb_loss)

        return total_loss / num_batches, total_mse_loss / num_batches, total_limb_loss / num_batches, total_kl_loss / num_batches

    def evaluate(self, X_val, y_val):
        X_val, y_val = self.preprocess(X_val, y_val)
        val_losses = self.process_batch(X_val, y_val, training=False)
        return val_losses
    
    def evaluate_missing(self, X_val, y_val):
        mask = np.isnan(X_val).reshape(X_val.shape[0], -1)
        X_val, y_val = self.preprocess(X_val, y_val)

        loss1, loss2 = 0, 0
        dist1, dist2 = 0, 0
        num_batches = 0
        for i in range(0, len(X_val), self.batch_size):
            X_batch = X_val[i:i + self.batch_size]
            y_batch = y_val[i:i + self.batch_size]
            mask_batch = mask[i:i + self.batch_size]
            num_batches += 1
            y_pred, _, _ = self.autoencoder(X_batch, training=False)
            missing_mse_loss, ns_mse_loss = self.MSE_missing(y_batch, y_pred, mask_batch)
            missing_dist, nm_dist = self.compute_dist(y_batch, y_pred, mask_batch)
            loss1 += np.mean(missing_mse_loss)
            loss2 += np.mean(ns_mse_loss)
            dist1 += missing_dist
            dist2 += nm_dist
        return loss1 / num_batches, loss2 / num_batches, dist1 / num_batches, dist2 / num_batches
    
    def compute_dist(self, y_batch, y_pred, mask):
        mask = mask.reshape(-1, 16, 3).any(axis=-1)
        y_batch = y_batch.reshape(-1, 16, 3)
        y_pred = y_pred.numpy().reshape(-1, 16, 3)

        dist = np.linalg.norm(y_batch - y_pred, axis=-1)
        dist_missing = dist[mask]
        dist_nm = dist[~mask]

        missing_dist = np.median(dist_missing)
        nm_dist = np.median(dist_nm)

        return missing_dist, nm_dist

    def MSE_missing(self, y_batch, y_pred, mask):
        y_batch = tf.convert_to_tensor(y_batch, tf.float32)

        y_true_missing = tf.boolean_mask(y_batch, mask)
        y_pred_missing = tf.boolean_mask(y_pred, mask)

        y_true_nm = tf.boolean_mask(y_batch, ~mask)
        y_pred_nm = tf.boolean_mask(y_pred, ~mask)

        missing_mse = tf.reduce_mean(tf.square(y_true_missing - y_pred_missing))
        nm_mse = tf.reduce_mean(tf.square(y_true_nm - y_pred_nm))

        return missing_mse.numpy(), nm_mse.numpy()

    def predict(self, X):
        normalized_data = self.poseprocessor.normalize(X)
        normalized_data = self.poseprocessor.replace_nan(normalized_data)
        with tf.device('/GPU:0'):
            # TODO: The first way is much slower, why?
            # pred, _, _ = self.autoencoder.predict(normalized_data.reshape(normalized_data.shape[0], -1), verbose=0)
            pred, _, _ = self.autoencoder(normalized_data.reshape(normalized_data.shape[0], -1))
        pred = pred.numpy().reshape(-1, 16, 3)

        res = self.poseprocessor.denormalize(pred, X)

        return res
    

class PoseProcessor:
    def __init__(self, bodyparts, original_point='spinemid', x_point='neck', xz_point='tailbase', scale_length=1):
        self.bp_dict = {bp: i for i, bp in enumerate(bodyparts)}

        self.op_idx = self.bp_dict[original_point]
        self.x_idx = self.bp_dict[x_point]
        self.xz_idx = self.bp_dict[xz_point]
        self.scale_length = scale_length

    @staticmethod
    def construct_rotation_matrix(original_point, x_point, xz_point):
        new_x_axis = x_point - original_point
        new_z_axis = -orthogonalize_vector(xz_point - original_point, new_x_axis)
        new_y_axis = np.cross(new_x_axis, new_z_axis)
        
        R = np.vstack([new_x_axis, new_y_axis, new_z_axis])
        R /= np.linalg.norm(R, axis=1)[:, None]
        return R.T

    def normalize(self, all_poses_3d):
        normalized_poses = np.zeros_like(all_poses_3d)
        for i, pose in enumerate(all_poses_3d):
            R = PoseProcessor.construct_rotation_matrix(pose[self.op_idx], pose[self.x_idx], pose[self.xz_idx])
            rotated_pose = (pose - pose[self.op_idx]) @ R
            normalized_poses[i] = rotated_pose/np.linalg.norm(pose[self.x_idx] - pose[self.op_idx])*self.scale_length

        return normalized_poses

    def denormalize(self, normalized_poses, original_poses):
        denormalized_poses = np.zeros_like(normalized_poses)
        for i, (normalized_pose, original_pose) in enumerate(zip(normalized_poses, original_poses)):
            R = PoseProcessor.construct_rotation_matrix(original_pose[self.op_idx], original_pose[self.x_idx], original_pose[self.xz_idx])
            scale_factor = np.linalg.norm(original_pose[self.x_idx] - original_pose[self.op_idx]) / self.scale_length
            rescaled_pose = normalized_pose * scale_factor

            denormalized_pose = rescaled_pose @ R.T + original_pose[self.op_idx]
            denormalized_poses[i] = denormalized_pose

        return denormalized_poses

    def mask(self, pose, mask_list):
        masked_pose = pose.copy()
        for bp in mask_list:
            bp_idx = self.bp_dict[bp]
            masked_pose[:, bp_idx] = np.nan
        return masked_pose
    
    @staticmethod
    def replace_nan(pose):
        filled_pose = np.nan_to_num(pose, nan=0.0, copy=True)
        return filled_pose
    
    
def generate_masked_data(data, bodyparts):
    preprocessor = PoseProcessor(bodyparts=bodyparts, original_point='spinemid', x_point='neck', xz_point='tailbase', scale_length=6)
    normalized_data = preprocessor.normalize(data)

    masked_data, gt_data = [], []
    for r in range(1, 4):
        for mask_list in itertools.combinations(bodyparts, r):
            masked_data.append(preprocessor.mask(normalized_data, mask_list))
            gt_data.append(normalized_data)

    masked_data = np.concatenate(masked_data, axis=0)
    gt_data = np.concatenate(gt_data, axis=0)

    return masked_data, gt_data