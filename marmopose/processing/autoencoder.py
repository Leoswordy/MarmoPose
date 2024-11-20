import time
import logging
import itertools

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset

logger = logging.getLogger(__name__)


class DAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], latent_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        z = self.encoder(x)
        return z


class DAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        dims = [latent_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        x_recon = self.decoder(z)
        return x_recon


class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], latent_dim=32):
        super().__init__()
        self.encoder = DAEEncoder(input_dim, hidden_dims, latent_dim)
        self.decoder = DAEDecoder(latent_dim, hidden_dims[::-1], input_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon
    

class DaeTrainer:
    def __init__(self, model, batch_size=64, bodyparts=None, skeleton_constraints=None):
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
            Preprocessed X, y and mask tensors.
        """
        assert X.shape == y.shape
        n_samples = X.shape[0]
        mask = np.isnan(X).astype(np.float32)  # 1 where data is missing

        X = self.poseprocessor.replace_nan(X)
        X = X.reshape(n_samples, -1)
        y = y.reshape(n_samples, -1)
        mask = mask.reshape(n_samples, -1)

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

    def build_dataloader(self, original_data, val_ratio=0.1):
        masked_data, gt_data = self.mask_data(original_data)
        X, y, mask = self.preprocess(masked_data, gt_data)

        dataset = PoseDataset(X, y, mask)
        n_samples = len(dataset)
        n_val = int(n_samples * val_ratio)
        n_train = n_samples - n_val

        train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
        logger.info(f'Split dataset into {len(train_dataset)} for train and {len(val_dataset)} for val using seed 42')

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def loss_fn(self, y_pred, y_true, mask):
        mse_loss = ((y_pred - y_true) ** 2 * mask).sum() / mask.sum()
        limb_loss = self.compute_limb_loss(y_pred)
        total_loss = mse_loss + limb_loss
        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'limb_loss': limb_loss
        }
    
    def compute_limb_loss(self, y_pred):
        y_pred = y_pred.view(-1, len(self.bodyparts), 3)  # Adjust dimensions as necessary

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
            all_losses = {'total_loss': 0, 'mse_loss': 0, 'limb_loss': 0}
            self.model.train()
            for X_batch, y_batch, mask_batch in self.train_dataloader:
                step_losses = self.train_step(X_batch, y_batch, mask_batch, optimizer)
                for k, v in step_losses.items():
                    all_losses[k] += v.item()
            num_batches = len(self.train_dataloader)
            avg_losses = {key: value / num_batches for key, value in all_losses.items()}
            loss_str = " | ".join([f"{key}: {value:.4f}" for key, value in avg_losses.items()])
            end_time = time.time()

            logger.info(f"Epoch {epoch+1}/{epochs}: Train: {loss_str}, Time: {end_time - start_time:.2f}s")

            val_loss = self.evaluate(self.val_dataloader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_path:
                    torch.save(self.model, save_path)
                    logger.info(f"Model saved to {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1} with val loss {val_loss:.4f}")
                    break
    
    def train_step(self, X_batch, y_batch, mask_batch, optimizer):
        X_batch, y_batch, mask_batch = X_batch.to(self.device), y_batch.to(self.device), mask_batch.to(self.device)
        optimizer.zero_grad()

        y_pred = self.model(X_batch)
        all_losses = self.loss_fn(y_pred, y_batch, mask_batch)

        all_losses['total_loss'].backward()
        optimizer.step()

        return all_losses
    
    def evaluate(self, val_dataloader):
        self.model.eval()

        with torch.no_grad():
            all_losses = {'total_loss': 0, 'mse_loss': 0, 'limb_loss': 0}
            for X_batch, y_batch, mask_batch in val_dataloader:
                X_batch, y_batch, mask_batch = X_batch.to(self.device), y_batch.to(self.device), mask_batch.to(self.device)
                y_pred = self.model(X_batch)
                step_losses = self.loss_fn(y_pred, y_batch, mask_batch)
                for k, v in step_losses.items():
                    all_losses[k] += v.item()

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
            pred = self.model(X_flat)
        pred = pred.cpu().numpy().reshape(X.shape)

        denormalized_pred = self.poseprocessor.denormalize(pred, X)
        return denormalized_pred


class PoseDataset(Dataset):
    def __init__(self, X, y, mask):
        self.X = X
        self.y = y
        self.mask = mask

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mask[idx]
    

class PoseProcessor:
    def __init__(self, bodyparts, original_point='spinemid', x_point='neck', xz_point='tailbase', scale_length=1):
        self.bodyparts = bodyparts
        self.bp_dict = {bp: i for i, bp in enumerate(bodyparts)}

        self.op_idx = self.bp_dict[original_point]
        self.x_idx = self.bp_dict[x_point]
        self.xz_idx = self.bp_dict[xz_point]
        self.scale_length = scale_length
    
    @staticmethod
    def orthogonalize_vector(a, b):
        return a - np.dot(a, b) / np.dot(b, b) * b

    @staticmethod
    def construct_rotation_matrix(original_point, x_point, xz_point):
        new_x_axis = x_point - original_point
        new_x_axis /= np.linalg.norm(new_x_axis)

        a = xz_point - original_point
        new_z_axis = PoseProcessor.orthogonalize_vector(a, new_x_axis)
        new_z_axis /= np.linalg.norm(new_z_axis)

        new_y_axis = np.cross(new_z_axis, new_x_axis)
        new_y_axis /= np.linalg.norm(new_y_axis)

        R = np.stack([new_x_axis, new_y_axis, new_z_axis], axis=1)  # Shape (3, 3)
        return R
    
    def normalize(self, all_poses_3d):
        normalized_poses = np.zeros_like(all_poses_3d)
        for i, pose in enumerate(all_poses_3d):
            R = self.construct_rotation_matrix(pose[self.op_idx], pose[self.x_idx], pose[self.xz_idx])
            rotated_pose = (pose - pose[self.op_idx]) @ R
            scale = np.linalg.norm(pose[self.x_idx] - pose[self.op_idx])
            normalized_poses[i] = rotated_pose / scale * self.scale_length
        return normalized_poses
    
    def denormalize(self, normalized_poses, original_poses):
        denormalized_poses = np.zeros_like(normalized_poses)
        for i, (normalized_pose, original_pose) in enumerate(zip(normalized_poses, original_poses)):
            R = self.construct_rotation_matrix(original_pose[self.op_idx], original_pose[self.x_idx], original_pose[self.xz_idx])
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
    