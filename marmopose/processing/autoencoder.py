import time
import logging
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from marmopose.utils.helpers import orthogonalize_vector

logger = logging.getLogger(__name__)


class DaeEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        intermediate_dim = hidden_dim * 2

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, intermediate_dim)
        self.fc31 = nn.Linear(intermediate_dim, latent_dim)  # z_mean
        self.fc32 = nn.Linear(intermediate_dim, latent_dim)  # z_log_var
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # z_mean, z_log_var
    

class DaeDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        intermediate_dim = hidden_dim * 2

        self.fc1 = nn.Linear(latent_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


class DenoisingAutoencoder(nn.Module):
    def __init__ (self, input_dim, hidden_dim=64, latent_dim=20):
        super().__init__()

        self.encoder = DaeEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = DaeDecoder(latent_dim, hidden_dim, input_dim)
    
    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5*z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps*std
    
    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var)
        return self.decoder(z), z_mean, z_log_var


class DaeTrainer:
    def __init__(self, model, batch_size=4096, bodyparts=None, skeleton_constraints=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"DAE device: {self.device}")
    
        self.model = model.to(self.device)
        self.batch_size = batch_size

        self.poseprocessor = PoseProcessor(bodyparts=bodyparts, original_point='spinemid', x_point='neck', xz_point='tailbase', scale_length=6)
        self.skeleton_constraints = skeleton_constraints
        self.bodyparts = bodyparts
    
    def mask_data(self, original_data):
        masked_data, gt_data = self.poseprocessor.generate_masked_data(original_data)
        return masked_data, gt_data

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
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def build_dataloader(self, original_data, val_ratio=0.1):
        masked_data, gt_data = self.mask_data(original_data)
        X, y = self.preprocess(masked_data, gt_data)

        dataset = TensorDataset(X, y)

        train_dataset, val_dataset = random_split(dataset, [1-val_ratio, val_ratio], torch.Generator().manual_seed(42))
        logger.info(f'Split dataset into {len(train_dataset)} for train and {len(val_dataset)} for val using seed 42')

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def loss_fn(self, y_pred, y_true, z_mean, z_log_var):
        mse_loss = F.mse_loss(y_pred, y_true)
        kl_loss = self.compute_kl_loss(z_mean, z_log_var)
        limb_loss = self.compute_limb_loss(y_pred)

        total_loss = mse_loss + kl_loss + limb_loss
        # total_loss = mse_loss + kl_loss
        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'kl_loss': kl_loss,
            'limb_loss': limb_loss
        }

    @staticmethod
    def compute_kl_loss(z_mean, z_log_var):
        kl_loss = 1 + z_log_var - z_mean.pow(2) - z_log_var.exp()
        kl_loss = torch.sum(kl_loss, dim=-1) * -0.5
        return kl_loss.mean()

    def compute_limb_loss(self, y_pred):
        y_pred = y_pred.view(-1, 16, 3)  # Adjust dimensions as necessary

        errors = []
        for (bp1, bp2), expected_length in self.skeleton_constraints:
            actual_lengths = torch.norm(y_pred[:, bp1] - y_pred[:, bp2], dim=1)
            errors.append(torch.abs(actual_lengths - expected_length / 10))

        errors_stacked = torch.stack(errors, dim=0)
        return torch.sum(errors_stacked, dim=0).mean()
    
    def train(self, save_path=None, epochs=100, lr=0.001, patience=10):
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        best_val_loss = np.inf
        patience_counter = 0

        for epoch in range(epochs):
            start_time = time.time()
            all_losses = {'total_loss': 0, 'mse_loss': 0, 'kl_loss': 0, 'limb_loss': 0}
            for X_batch, y_batch in self.train_dataloader:
                step_losses = self.train_step(X_batch, y_batch, optimizer)
                for k, v in step_losses.items():
                    all_losses[k] += v
            
            num_batches = len(self.train_dataloader)
            avg_losses = {key: value / num_batches for key, value in all_losses.items()}
            loss_str = " | ".join([f"{key}: {value:.4f}" for key, value in avg_losses.items()])
            end_time = time.time()

            logger.info(f"********* Epoch {epoch}/{epochs} *********: Train: {loss_str}, Time: {end_time - start_time:.2f}s")

            val_loss = self.evaluate(self.val_dataloader)  # Ensure this also uses .to(device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch} with val loss {val_loss:.4f}")
                    break
            
        if save_path:
            torch.save(self.model, save_path)
            logger.info(f"Model saved to {save_path}")
    
    def train_step(self, X_batch, y_batch, optimizer):
        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
        optimizer.zero_grad()

        y_pred, z_mean, z_log_var = self.model(X_batch)
        all_losses = self.loss_fn(y_pred, y_batch, z_mean, z_log_var)

        all_losses['total_loss'].backward()
        optimizer.step()

        return all_losses
    
    def evaluate(self, val_dataloader):
        self.model.eval()

        with torch.no_grad():
            all_losses = {'total_loss': 0, 'mse_loss': 0, 'kl_loss': 0, 'limb_loss': 0}
            for X_batch, y_batch in val_dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred, z_mean, z_log_var = self.model(X_batch)
                step_losses = self.loss_fn(y_pred, y_batch, z_mean, z_log_var)
                for k, v in step_losses.items():
                    all_losses[k] += v
            
            num_batches = len(val_dataloader)
            avg_losses = {key: value / num_batches for key, value in all_losses.items()}
            loss_str = " | ".join([f"{key}: {value:.4f}" for key, value in avg_losses.items()])

            logger.info(f"Validation: {loss_str}")
        
        return avg_losses['total_loss']
    
    def predict(self, X):
        X_normalized = self.poseprocessor.normalize(X)
        X_normalized = self.poseprocessor.replace_nan(X_normalized)
        X_flat = torch.tensor(X_normalized.reshape(len(X_normalized), -1), dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            pred, _, _ = self.model(X_flat)
        
        denormalized_pred = self.poseprocessor.denormalize(pred.cpu().numpy().reshape(X.shape), X)

        return denormalized_pred


class PoseProcessor:
    def __init__(self, bodyparts, original_point='spinemid', x_point='neck', xz_point='tailbase', scale_length=1):
        self.bodyparts = bodyparts
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
    
    def generate_masked_data(self, data):
        normalized_data = self.normalize(data)

        masked_data, gt_data = [], []
        for r in range(1, 4):
            for mask_list in itertools.combinations(self.bodyparts, r):
                masked_data.append(self.mask(normalized_data, mask_list))
                gt_data.append(normalized_data)

        masked_data = np.concatenate(masked_data, axis=0)
        gt_data = np.concatenate(gt_data, axis=0)

        return masked_data, gt_data