#!/usr/bin/env python3
"""
Temporal Action Segmentation with Multi-Stage Model and Foreground-Aware Refinement

This module implements a multi-stage temporal convolutional network for action segmentation
with optional foreground-aware refinement and segment boundary learning components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
import time


class MultiStageModel(nn.Module):
    """
    Multi-stage temporal convolutional network for action segmentation.
    
    Args:
        num_stages (int): Number of refinement stages
        num_layers (int): Number of dilated residual layers per stage
        num_f_maps (int): Number of feature maps in intermediate layers
        dim (int): Input feature dimension
        num_classes (int): Number of action classes
        use_faar (bool): Whether to use Foreground-Aware Refinement
    """
    
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes, use_faar=True):
        super(MultiStageModel, self).__init__()
        
        # First stage processes raw features
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes, use_faar=use_faar)
        
        # Subsequent stages refine predictions from previous stages
        self.stages = nn.ModuleList([
            copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes, use_faar=use_faar)) 
            for _ in range(num_stages - 1)
        ])

    def forward(self, x, fg_bg_feat, mask, k=1, fg_weight=1):
        """
        Forward pass through multi-stage model.
        
        Args:
            x (torch.Tensor): Input features [batch_size, dim, num_frames]
            fg_bg_feat (torch.Tensor): Foreground/background features for refinement
            mask (torch.Tensor): Valid frame mask
            k (int): Number of top predictions to consider for refinement
            fg_weight (float): Weight for foreground refinement
            
        Returns:
            torch.Tensor: Multi-stage predictions [num_stages, batch_size, num_classes, num_frames]
        """
        # First stage
        out = self.stage1(x, fg_bg_feat, mask, k=k, fg_weight=fg_weight)
        outputs = out.unsqueeze(0)
        
        # Refinement stages
        for stage in self.stages:
            # Use softmax predictions from previous stage as input
            out = stage(F.softmax(out, dim=1) * mask[:, 0:1, :], fg_bg_feat, mask, k=k, fg_weight=fg_weight)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
            
        return outputs


class SingleStageModel(nn.Module):
    """
    Single stage of the temporal convolutional network.
    
    Args:
        num_layers (int): Number of dilated residual layers
        num_f_maps (int): Number of feature maps
        dim (int): Input feature dimension
        num_classes (int): Number of output classes
        use_faar (bool): Whether to use Foreground-Aware Refinement
    """
    
    def __init__(self, num_layers, num_f_maps, dim, num_classes, use_faar=False):
        super(SingleStageModel, self).__init__()
        
        # Initial 1x1 convolution to adjust feature dimension
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        
        # Stack of dilated residual layers with exponentially increasing dilation
        self.layers = nn.ModuleList([
            copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) 
            for i in range(num_layers)
        ])
        
        # Output layer
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        
        # Optional foreground-aware refinement
        self.use_faar = use_faar
        if self.use_faar:
            self.faar = ForegroundAwareRefinementModel(512, num_classes)

    def forward(self, x, fg_bg_feat, mask, k=1, fg_weight=1):
        """Forward pass through single stage."""
        # Initial feature processing
        out = self.conv_1x1(x)
        
        # Dilated residual layers
        for layer in self.layers:
            out = layer(out, mask)
        
        # Generate class predictions
        out = self.conv_out(out)
        
        # Apply foreground-aware refinement if enabled
        if self.use_faar:
            out = self.faar(fg_bg_feat, out, k=k, fg_weight=fg_weight)
        
        # Apply mask to remove invalid frames
        out = out * mask[:, 0:1, :]
        
        return out


class DilatedResidualLayer(nn.Module):
    """
    Dilated residual layer for temporal modeling.
    
    Args:
        dilation (int): Dilation rate for temporal convolution
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask=None):
        """Forward pass with residual connection."""
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        
        # Add residual connection
        out = x + out
        
        # Apply mask if provided
        if mask is not None:
            out = out * mask[:, 0:1, :]
            
        return out


class ForegroundAwareRefinementModel(nn.Module):
    """
    Foreground-Aware Refinement Module that refines predictions using
    foreground/background feature information.
    
    Args:
        dim (int): Feature dimension
        num_classes (int): Number of action classes
    """
    
    def __init__(self, dim, num_classes):
        super(ForegroundAwareRefinementModel, self).__init__()
        self.dim = dim
        self.num_classes = num_classes
        
        # Refinement network parameters
        num_f_maps = 64
        num_layers = 3
        
        # Feature processing layers
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([
            copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) 
            for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, fg_bg_feat, raw_logits, k=1, fg_weight=1):
        """
        Refine predictions using foreground features from top-k predictions.
        
        Args:
            fg_bg_feat (torch.Tensor): Foreground/background features [batch_size, num_class, feat_dim, num_frames]
            raw_logits (torch.Tensor): Raw predictions to refine [batch_size, num_classes, num_frames]
            k (int): Number of top predictions to consider
            fg_weight (float): Weight for combining refined and raw predictions
            
        Returns:
            torch.Tensor: Refined predictions
        """
        # Get top-k predictions
        topk_values, topk_indices = torch.topk(raw_logits, k, dim=1)
        
        # Rearrange features to extract top-k foreground features
        features_permuted = fg_bg_feat.permute(0, 3, 1, 2)  # [batch, frames, classes, feat_dim]
        topk_indices_transposed = topk_indices.permute(0, 2, 1)  # [batch, frames, k]
        
        # Expand indices for gathering
        topk_indices_expanded = topk_indices_transposed.unsqueeze(-1)  # [batch, frames, k, 1]
        feat_dim = features_permuted.shape[3]
        topk_indices_expanded = topk_indices_expanded.expand(-1, -1, -1, feat_dim)
        
        # Extract top-k features
        topk_features = torch.gather(features_permuted, dim=2, index=topk_indices_expanded)
        fg_feat = topk_features[..., :self.dim]  # Use only foreground features
        
        # Process features through refinement network
        x = fg_feat.permute(0, 2, 3, 1)  # [batch, k, feat_dim, frames]
        x = x.view(-1, *x.shape[2:])  # [batch*k, feat_dim, frames]
        
        # Apply refinement layers
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask=None)
        out = self.conv_out(out)
        
        # Reshape and aggregate
        out = out.view(-1, k, *out.shape[1:])  # [batch, k, num_classes, frames]
        out = out.permute(0, 3, 1, 2)  # [batch, frames, k, num_classes]
        
        # Weighted combination of top-k predictions
        topk_weights = F.softmax(topk_values, dim=1)  # [batch, k, frames]
        weighted_out = torch.einsum('bfkc,bkf->bfc', out, topk_weights)
        weighted_out = weighted_out.permute(0, 2, 1)  # [batch, num_classes, frames]
        
        # Combine with raw predictions
        refined_out = (raw_logits + fg_weight * weighted_out) / (1 + fg_weight)
        
        return refined_out


class SegmentBoundaryLearning(nn.Module):
    """
    Segment Boundary Learning module for detecting action boundaries.
    
    Args:
        feat_dim (int): Feature dimension
        out_channels (int): Number of output channels
        kernel_size (int): Convolution kernel size
        window_size (int): Size of temporal window
        margin (int): Margin around boundaries
    """
    
    def __init__(self, feat_dim, out_channels, kernel_size, window_size, margin):
        super(SegmentBoundaryLearning, self).__init__()
        
        # Separate convolutions for left and right contexts
        self.conv1d_left = nn.Conv2d(in_channels=1, out_channels=5, 
                                    kernel_size=kernel_size, 
                                    padding=(kernel_size - 1) // 2)
        self.conv1d_right = nn.Conv2d(in_channels=1, out_channels=5,
                                     kernel_size=kernel_size, 
                                     padding=(kernel_size - 1) // 2)
        
        # Output projection
        self.fc = nn.Linear(5, 1)
        self.window_size = window_size
        self.margin = margin

    def forward(self, features):
        """
        Detect segment boundaries in feature sequence.
        
        Args:
            features (torch.Tensor): Input features [batch_size, n_frames, feat_dim]
            
        Returns:
            torch.Tensor: Boundary-aware features [batch_size, n_frames, feat_dim]
        """
        batch_size, n_frames, feat_dim = features.shape

        # Pad features for windowing
        padded_features = F.pad(features, (0, 0, self.window_size, self.window_size))
        
        # Create left and right context windows
        left_grouped_features = self._group_video_features(
            padded_features[:, :-self.window_size-self.margin, :], 
            self.window_size - self.margin + 1
        )
        right_grouped_features = self._group_video_features(
            padded_features[:, self.window_size+self.margin:, :], 
            self.window_size - self.margin + 1
        )

        # Reshape for convolution
        left_grouped_features = left_grouped_features.view(-1, 1, *left_grouped_features.shape[2:])        
        right_grouped_features = right_grouped_features.view(-1, 1, *right_grouped_features.shape[2:]) 

        # Process left and right contexts
        x_left = self.conv1d_left(left_grouped_features.transpose(-1, -2))
        x_right = self.conv1d_right(right_grouped_features.transpose(-1, -2))
        
        # Combine and process
        x = torch.cat((x_left, x_right), dim=-1)
        x = x.permute(0, 2, 3, 1)
        x = self.fc(x).squeeze(-1)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = x.view(batch_size, -1, feat_dim)
        
        return x
    
    def _group_video_features(self, video_features, window_size):
        """Group video features into sliding windows."""
        batch_size, n_frames, dim = video_features.shape
        output_shape = (batch_size, n_frames - window_size + 1, window_size, dim)
        grouped_features = torch.zeros(output_shape, dtype=video_features.dtype, device=video_features.device)
        
        for t in range(n_frames - window_size + 1):
            grouped_features[:, t, :, :] = video_features[:, t:t + window_size, :]
        
        return grouped_features


class DualEncoder(nn.Module):
    """
    Dual encoder for foreground and background feature processing.
    
    Args:
        dim (int): Feature dimension
        use_att (bool): Whether to use attention mechanism
    """
    
    def __init__(self, dim, use_att=False):
        super(DualEncoder, self).__init__()
        self.dim = dim
        self.fg_model = self._build_reconstruct_model(dim)
        self.bg_model = self._build_reconstruct_model(dim)
        self.use_att = use_att
        
        if self.use_att:
            self.cross = nn.Linear(dim, dim)
    
    def _build_reconstruct_model(self, dim):
        """Build reconstruction model for feature processing."""
        return nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim // 2),
        )
        
    def forward(self, fg, bg):
        """
        Process foreground and background features.
        
        Args:
            fg (torch.Tensor): Foreground features
            bg (torch.Tensor): Background features
            
        Returns:
            torch.Tensor: Combined processed features
        """
        # Process through separate encoders
        fg = fg.permute(0, 2, 1)
        bg = bg.permute(0, 2, 1)
        
        fg_processed = self.fg_model(fg)
        bg_processed = self.bg_model(bg)
        
        # Combine features
        concat = torch.cat([fg_processed, bg_processed], dim=-1)
        
        # Apply attention if enabled
        if self.use_att:
            att_output = self.cross(concat)
            concat = concat + att_output
            
        return concat.permute(0, 2, 1)


def mixup_data(data, alpha=0.8):
    """
    Apply mixup augmentation to background features.
    
    Args:
        data (torch.Tensor): Input data [batch_size, feature_dim, n_frames]
        alpha (float): Mixup parameter controlling interpolation strength
        
    Returns:
        torch.Tensor: Mixed data
    """
    batch_size, feature_dim, n_frames = data.shape  
    assert feature_dim == 2048, "Expected feature dimension of 2048"
    
    # Generate random weights for mixing
    random_weights = torch.rand(batch_size, n_frames, batch_size, device=data.device)
    random_weights = random_weights / torch.clamp(torch.sum(random_weights, dim=-1, keepdim=True), min=1e-10)
    
    # Create mixing mask
    mask = torch.rand((batch_size, n_frames, 1), device=data.device) 
    eye_mask = torch.eye(batch_size, device=data.device).unsqueeze(1).repeat(1, n_frames, 1)
    
    # Apply selective mixing
    weights = (eye_mask * (mask <= alpha)) + (0.8 * eye_mask + 0.2 * random_weights) * (mask > alpha)
    mixed_data = torch.einsum('bfB,Bdf->bdf', weights, data)
    
    return mixed_data


class Trainer:
    """
    Trainer class for temporal action segmentation model.
    
    Args:
        num_blocks (int): Number of stages in multi-stage model
        num_layers (int): Number of layers per stage
        num_f_maps (int): Number of feature maps
        dim (int): Input feature dimension
        num_classes (int): Number of action classes
        logger: Logger instance
        prev_model_path (str): Path to previous model for initialization
        dual_use_att (bool): Whether to use attention in dual encoder
        use_sbl (bool): Whether to use segment boundary learning
        use_fbfc (bool): Whether to use foreground-background feature composition
        use_faar (bool): Whether to use foreground-aware refinement
    """
    
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes, logger, 
                 prev_model_path=None, dual_use_att=False, use_sbl=False, use_fbfc=False, use_faar=True):
        
        # Initialize main model
        self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes, use_faar)
        
        # Initialize optional components
        self.use_sbl = use_sbl
        self.use_fbfc = use_fbfc
        self.use_faar = use_faar
        
        if self.use_sbl:
            self.sbl = SegmentBoundaryLearning(dim, num_f_maps, kernel_size=3, window_size=5, margin=2)
            
        if self.use_fbfc:
            self.dual_model = DualEncoder(dim, use_att=dual_use_att)
        
        # Loss functions
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        
        # Training parameters
        self.num_classes = num_classes
        self.logger = logger
        self.dim = dim
        self.prev_model_path = prev_model_path

    def load_model(self, model_path):
        """Load model weights from checkpoint."""
        states_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(states_dict['action_model'], strict=False)
        
        if self.use_sbl and 'sbl' in states_dict:
            self.logger.info("Loading SBL parameters")
            self.sbl.load_state_dict(states_dict['sbl'])
            
        if self.use_fbfc and 'dual_model' in states_dict:
            self.logger.info("Loading dual model parameters")
            self.dual_model.load_state_dict(states_dict['dual_model'])
            
        self.logger.info(f"Loaded model from {model_path}")
    
    def save_model(self, save_path):
        """Save model weights to checkpoint."""
        to_save = {'action_model': self.model.state_dict()}
        
        if self.use_sbl:
            to_save['sbl'] = self.sbl.state_dict()
        if self.use_fbfc:
            to_save['dual_model'] = self.dual_model.state_dict()
            
        torch.save(to_save, save_path)

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device):
        """
        Train the model.
        
        Args:
            save_dir (str): Directory to save model checkpoints
            batch_gen: Batch generator for training data
            num_epochs (int): Number of training epochs
            batch_size (int): Batch size
            learning_rate (float): Learning rate
            device: Training device (CPU/GPU)
        """
        # Load previous model if specified
        if self.prev_model_path:
            self.load_model(self.prev_model_path)
            
        # Move models to device
        self.model.to(device)
        self.model.train()
        
        # Prepare optimizer
        trainable_params = [{'params': self.model.parameters()}]
        
        if self.use_sbl:
            self.sbl.to(device)
            self.sbl.eval()  # Keep SBL in eval mode during training
            
        if self.use_fbfc:
            self.dual_model.to(device)
            self.dual_model.train()
            trainable_params.append({'params': self.dual_model.parameters()})

        optimizer = optim.Adam(trainable_params, lr=learning_rate)
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_rec_loss = 0
            epoch_sbl_loss = 0
            correct = 0
            total = 0
            
            while batch_gen.has_next():
                batch_dict = batch_gen.next_batch(batch_size)
                batch_input = batch_dict['batch_input']
                batch_target = batch_dict['batch_target']
                mask = batch_dict['mask']
                batch_fg_bg_feat = batch_dict['batch_fg_bg_feat']
                batch_boundary_label = batch_dict['batch_boundary_label']
                
                # Limit sequence length for memory efficiency
                max_frames = 10000 if epoch < 20 else 10000
                if batch_input.shape[-1] > max_frames:
                    batch_input = batch_input[..., :max_frames]
                    batch_target = batch_target[..., :max_frames]
                    mask = mask[..., :max_frames]
                    batch_fg_bg_feat = batch_fg_bg_feat[..., :max_frames]
                    batch_boundary_label = batch_boundary_label[..., :max_frames]

                # Move to device
                batch_input = batch_input.to(device)
                batch_target = batch_target.to(device)
                batch_fg_bg_feat = batch_fg_bg_feat.to(device)
                mask = mask.to(device)
                batch_boundary_label = batch_boundary_label.to(device)

                optimizer.zero_grad()
                
                # Extract original input features
                batch_input_ori = batch_input[:, :self.dim, :]
                batch_final_input = batch_input_ori
                
                # Apply segment boundary learning if enabled
                sbl_loss = torch.tensor(0.0, device=device)
                if self.use_sbl:
                    batch_sbl_output = self.sbl(batch_input_ori.transpose(2, 1)).transpose(2, 1)
                    sbl_loss = self.mse(batch_input_ori, batch_sbl_output).mean(dim=1)
                    sbl_loss = (sbl_loss * (batch_boundary_label == 0)).sum() / (batch_boundary_label == 0).sum()
                    
                    # Use SBL output at boundaries
                    batch_final_input = (batch_input_ori * (batch_boundary_label == 0)[:, None, :] + 
                                       batch_sbl_output.data * (batch_boundary_label == 1)[:, None, :])

                # Prepare foreground-background features
                if len(batch_fg_bg_feat.size()) == 3:
                    batch_fg_bg_feat = batch_fg_bg_feat.unsqueeze(1).repeat(1, self.num_classes, 1, 1)

                # Set foreground weight based on training progress
                if not self.use_faar or epoch < num_epochs // 2:
                    fg_weight = 0  # No FAAR in first half of training
                else:
                    fg_lambda = np.random.rand()
                    fg_weight = fg_lambda / max(1 - fg_lambda, 1e-10)

                # Apply foreground-background feature composition if enabled
                reconstruct_loss = torch.tensor(0.0, device=device)
                if self.use_fbfc:
                    batch_input_fg = batch_input[:, self.dim:2*self.dim, :]
                    batch_input_bg = batch_input[:, 2*self.dim:, :]
                    
                    # Reconstruction loss
                    batch_output_comp = self.dual_model(batch_input_fg, batch_input_bg)
                    reconstruct_loss = self.mse(batch_input_ori, batch_output_comp).mean()
                    
                    # Apply mixup and use composed features
                    batch_input_bg_fbfc = mixup_data(batch_input_bg, alpha=0.95)
                    batch_output_fbfc = self.dual_model(batch_input_fg, batch_input_bg_fbfc)
                    batch_final_input = batch_output_fbfc.data

                # Forward pass through main model
                batch_final_input = batch_final_input.detach()  # Stop gradients from flowing to preprocessing
                predictions = self.model(batch_final_input, batch_fg_bg_feat, mask, k=1, fg_weight=fg_weight)

                # Compute main loss
                loss = torch.tensor(0.0, device=device)
                for p in predictions:
                    # Classification loss
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), 
                                  batch_target.view(-1))
                    
                    # Smoothness loss
                    smoothness_loss = torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), 
                               F.log_softmax(p.detach()[:, :, :-1], dim=1)), 
                        min=0, max=16
                    ) * mask[:, :, 1:]
                    loss += 0.15 * torch.mean(smoothness_loss)

                # Add auxiliary losses
                loss += sbl_loss + reconstruct_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()

                # Update statistics
                epoch_loss += loss.item()
                epoch_sbl_loss += sbl_loss.item()
                epoch_rec_loss += reconstruct_loss.item()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float() * mask[:, 0, :]).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            # Reset batch generator
            batch_gen.reset()
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_model(f"{save_dir}/epoch-{epoch + 1}.model")
                torch.save(optimizer.state_dict(), f"{save_dir}/epoch-{epoch + 1}.opt")
            
            # Log training progress
            accuracy = float(correct) / max(total, 1e-4)
            avg_loss = epoch_loss / len(batch_gen.list_of_examples)
            avg_sbl_loss = epoch_sbl_loss / len(batch_gen.list_of_examples)
            avg_rec_loss = epoch_rec_loss / len(batch_gen.list_of_examples)
            
            self.logger.info(
                f"[Epoch {epoch + 1}]: Loss = {avg_loss:.4f}, "
                f"SBL Loss = {avg_sbl_loss:.4f}, Reconstruct Loss = {avg_rec_loss:.4f}, "
                f"Accuracy = {accuracy:.4f}"
            )

    def predict_with_dataloader(self, model_dir, results_dir, batch_gen, vid_list_file, 
                              epoch, actions_dict, device, sample_rate, 
                              feature_transpose=False, map_delimiter=' '):
        """
        Generate predictions using a data loader.
        
        Args:
            model_dir (str): Directory containing model checkpoints
            results_dir (str): Directory to save prediction results
            batch_gen: Batch generator for test data
            vid_list_file (str): File containing list of video names
            epoch (int): Epoch number of model to load
            actions_dict (dict): Mapping from action names to class indices
            device: Inference device
            sample_rate (int): Frame sampling rate
            feature_transpose (bool): Whether to transpose features
            map_delimiter (str): Delimiter for output format
        """
        self.model.eval()
        
        with torch.no_grad():
            self.model.to(device)
            self.load_model(f"{model_dir}/epoch-{epoch}.model")
            
            # Load video list
            with open(vid_list_file, 'r') as f:
                list_of_vids = f.read().split('\n')[:-1]

            count = 0
            while batch_gen.has_next():
                vid = list_of_vids[count]
                count += 1
                
                # Get batch
                batch_dict = batch_gen.next_batch(batch_size=1)
                batch_input = batch_dict['batch_input'].to(device)
                batch_target = batch_dict['batch_target'].to(device)
                mask = batch_dict['mask'].to(device)
                batch_fg_bg_feat = batch_dict['batch_fg_bg_feat'].to(device)

                # Extract input features
                batch_input_ori = batch_input[:, :self.dim, :]
                
                # Prepare foreground-background features
                if len(batch_fg_bg_feat.size()) == 3:
                    batch_fg_bg_feat = batch_fg_bg_feat.unsqueeze(1).repeat(1, self.num_classes, 1, 1)

                # Generate predictions
                fg_lambda = 0.3
                fg_weight = fg_lambda / max(1 - fg_lambda, 1e-10)
                predictions = self.model(batch_input_ori, batch_fg_bg_feat, mask, k=3, fg_weight=fg_weight)
                
                # Convert to action labels
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                
                recognition = []
                for i in range(len(predicted)):
                    action_name = list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]
                    recognition.extend([action_name] * sample_rate)
                
                # Save results
                f_name = vid.split('/')[-1].split('.')[0]
                with open(f"{results_dir}/{f_name}", "w") as f:
                    f.write("### Frame level recognition: ###\n")
                    f.write(map_delimiter.join(recognition))