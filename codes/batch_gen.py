#!/usr/bin/env python3
"""
Batch generator for temporal action segmentation.

This module provides functionality for loading and batching video features and ground truth
labels for temporal action segmentation tasks. It supports various data augmentation techniques
including segment trimming, concatenation, and boundary learning.
"""

import torch
import numpy as np
import random
from collections import defaultdict
import os


def get_segment_points(sequence):
    """
    Get segment boundaries from an action sequence.
    
    Args:
        sequence (np.ndarray): Array of action labels
        
    Returns:
        list: List of frame indices where action changes occur
    """
    return (np.where(np.diff(sequence))[0] + 1).tolist()


def get_segment_points_with_actions(sequence):
    """
    Get segment boundaries along with the actions before and after each boundary.
    
    Args:
        sequence (np.ndarray): Array of action labels
        
    Returns:
        list: List of [boundary_point, prev_action, next_action] for each boundary
    """
    # Find points where the value changes
    points = np.where(np.diff(sequence))[0] + 1
    non_zero_index = np.nonzero(sequence)[0]
    segments = []
    
    for point in points:
        # Find previous and next actions
        prev_index = np.where(non_zero_index < point)[0]
        next_index = np.where(non_zero_index >= point)[0]
        
        prev_action = sequence[non_zero_index[prev_index[-1]]] if len(prev_index) > 0 else 0
        next_action = sequence[non_zero_index[next_index[0]]] if len(next_index) > 0 else 0
        
        segments.append([int(point), int(prev_action), int(next_action)])
    
    return segments


def trim_and_concat_features(*batch_feats, batch_target, max_num_point=None):
    """
    Trim and concatenate segments from multiple videos with multiple feature types.
    
    Args:
        *batch_feats: Variable number of feature arrays
        batch_target: List of target action sequences
        max_num_point: Maximum number of segmentation points per video
        
    Returns:
        tuple: (final_feats, final_target) where final_feats is a list of concatenated features
    """
    points_list = []
    video_list = []
    
    # Get segment points for each video
    for video_i, target in enumerate(batch_target):
        points = get_segment_points(target)
        if max_num_point is not None and len(points) > max_num_point:
            points = sorted(random.sample(points, k=max_num_point))
        points = [0] + points + [target.shape[0]]
        points_list.append(points)
        video_list.extend([video_i] * (len(points) - 1))

    random.shuffle(video_list)

    # Extract segments
    curr_segment = defaultdict(int)
    final_target = []
    final_feats = [[] for _ in batch_feats]

    for video in video_list:
        points = points_list[video]
        start, end = points[curr_segment[video]], points[curr_segment[video] + 1]
        final_target.append(batch_target[video][start:end])
        
        for i, batch_feat in enumerate(batch_feats):
            final_feats[i].append(batch_feat[video][:, start:end])
        
        curr_segment[video] += 1

    # Concatenate features
    final_feats = [[np.concatenate(feat_list, axis=-1)] for feat_list in final_feats]
    final_target = [np.concatenate(final_target, axis=0)]
    
    return (*final_feats, final_target)


def trim_and_concat_with_boundaries(*batch_feats, batch_target, batch_task_label, max_num_point=None):
    """
    Advanced trim and concatenate with boundary learning support.
    
    Args:
        *batch_feats: Variable number of feature arrays
        batch_target: List of target action sequences
        batch_task_label: List of task labels for each video
        max_num_point: Maximum number of segmentation points per video
        
    Returns:
        tuple: (final_feats, final_target, final_boundary_label)
    """
    if max_num_point == 0:
        # Simple concatenation without trimming
        final_feats = [[] for _ in batch_feats]
        final_target = []
        final_boundary_label = []
        
        for video_i in range(len(batch_target)):
            for i, batch_feat in enumerate(batch_feats):
                final_feats[i].append(batch_feat[video_i])
            final_target.append(batch_target[video_i])
            final_boundary_label.extend([0] * len(batch_target[video_i]))
        
        final_feats = [[np.concatenate(feat_list, axis=-1)] for feat_list in final_feats]
        final_target = [np.concatenate(final_target, axis=0)]
        final_boundary_label = [np.array(final_boundary_label)]
        
        return (*final_feats, final_target, final_boundary_label)
    
    # Get segment information for each video
    video_stack = {}
    for video_i, target in enumerate(batch_target):
        points = get_segment_points_with_actions(target)
        
        if max_num_point is not None and len(points) > max_num_point:
            points = sorted(random.sample(points, k=max_num_point))
        
        # Create segments
        segments = [[0, *points[0]]] if points else []
        for p, point in enumerate(points[1:]):
            segments.append([points[p][0], *point])
        segments.append([points[-1][0], len(target), 0, 0])
        
        video_stack[video_i] = segments
    
    # Randomly select initial video and build concatenated sequence
    curr_video = random.choice(range(len(batch_target)))
    final_target = []
    final_feats = [[] for _ in batch_feats]
    final_boundary_label = []
    is_switch = False
    
    while any(len(video_stack[video]) for video in video_stack):
        # Get next segment from current video
        segment = video_stack[curr_video].pop(0)
        start, end, curr_action, next_action = segment
        
        # Add segment to final sequences
        final_target.append(batch_target[curr_video][start:end])
        for i, batch_feat in enumerate(batch_feats):
            final_feats[i].append(batch_feat[curr_video][:, start:end])
        
        # Create boundary labels
        boundary_label = np.zeros([end - start])
        if is_switch:
            boundary_label[:5] = 1  # Mark beginning of switched segment
        
        # Decide whether to switch videos
        other_videos = [v for v in video_stack if v != curr_video and len(video_stack[v]) > 0]
        if other_videos:
            # Simple random switching (can be enhanced with continuation probability)
            switch_prob = 0.3  # Probability of switching to another video
            if np.random.rand() < switch_prob and len(video_stack[curr_video]) > 0:
                is_switch = False
            else:
                curr_video = random.choice(other_videos)
                is_switch = True
        else:
            is_switch = False
        
        # Mark end of segment if switching
        if is_switch:
            boundary_label[-5:] = 1
            
        final_boundary_label.extend(boundary_label.tolist())
    
    # Concatenate all features
    final_feats = [[np.concatenate(feat_list, axis=-1)] for feat_list in final_feats]
    final_target = [np.concatenate(final_target, axis=0)]
    final_boundary_label = [np.array(final_boundary_label)]
    
    return (*final_feats, final_target, final_boundary_label)


class BatchGenerator:
    """
    Batch generator for temporal action segmentation datasets.
    
    Args:
        num_classes (int): Number of action classes
        actions_dict (dict): Mapping from action names to class indices
        gt_path (str): Path to ground truth files
        features_path (str): Path to feature files
        sample_rate (int): Frame sampling rate
        feature_transpose (bool): Whether to transpose features
        num_video_concat (int): Number of videos to concatenate
        max_num_point (int): Maximum number of segment points per video
        fg_bg_features_path (str): Path to foreground/background features
        mode (str): 'train' or 'test' mode
    """
    
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate=1,
                 feature_transpose=False, num_video_concat=1, max_num_point=None,
                 fg_bg_features_path='', mode='train'):
        
        self.list_of_examples = []
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.feature_transpose = feature_transpose
        self.num_video_concat = num_video_concat
        self.max_num_point = max_num_point
        self.fg_bg_features_path = fg_bg_features_path
        self.load_fg_bg_feat = bool(fg_bg_features_path)
        self.mode = mode

    def reset(self):
        """Reset the batch generator and shuffle examples if in training mode."""
        self.index = 0
        if self.mode == 'train':
            random.shuffle(self.list_of_examples)

    def has_next(self):
        """Check if there are more batches available."""
        return self.index < len(self.list_of_examples)

    def read_data(self, vid_list_file, num_videos=1000000):
        """
        Read video list from file.
        
        Args:
            vid_list_file (str): Path to file containing video names
            num_videos (int): Maximum number of videos to load
        """
        with open(vid_list_file, 'r') as f:
            self.list_of_examples = f.read().split('\n')[:-1]
        
        if len(self.list_of_examples) > num_videos:
            self.list_of_examples = self.list_of_examples[:num_videos]
        
        if self.mode == 'train':
            random.shuffle(self.list_of_examples)

    def _load_video_data(self, vid):
        """
        Load features and ground truth for a single video.
        
        Args:
            vid (str): Video filename
            
        Returns:
            tuple: (features, classes, fg_bg_features)
        """
        # Load main features
        feature_file = os.path.join(self.features_path, vid.split('.')[0] + '.npy')
        features = np.load(feature_file)
        if self.feature_transpose:
            features = features.T
        
        # Load ground truth
        gt_file = os.path.join(self.gt_path, vid)
        with open(gt_file, 'r') as f:
            content = f.read().split('\n')[:-1]
        
        # Convert actions to class indices
        classes = np.zeros(min(features.shape[1], len(content)), dtype=np.int32)
        for i in range(len(classes)):
            classes[i] = self.actions_dict[content[i]]
        
        # Load foreground/background features if available
        fg_bg_features = None
        if self.load_fg_bg_feat:
            fg_bg_file = os.path.join(self.fg_bg_features_path, vid.split('.')[0] + '.npy')
            fg_bg_features = np.load(fg_bg_file)
            if self.feature_transpose:
                fg_bg_features = fg_bg_features.swapaxes(-2, -1)
            
            # For training with multitask data, select class-specific features
            if self.mode == 'train' and 'multitask' in vid:
                fg_bg_features = fg_bg_features[classes, :, np.arange(fg_bg_features.shape[-1])].T
        
        return features, classes, fg_bg_features

    def _create_tensors(self, batch_input, batch_target, batch_fg_bg_input, batch_boundary_label, batch_vid):
        """
        Convert batch data to PyTorch tensors.
        
        Args:
            batch_input: List of input features
            batch_target: List of target sequences
            batch_fg_bg_input: List of foreground/background features
            batch_boundary_label: List of boundary labels
            batch_vid: List of video names
            
        Returns:
            dict: Dictionary containing batched tensors
        """
        length_of_sequences = [len(target) for target in batch_target]
        max_length = max(length_of_sequences)
        
        # Initialize tensors
        batch_input_tensor = torch.zeros(len(batch_input), batch_input[0].shape[0], max_length, dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max_length, dtype=torch.long) * (-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max_length, dtype=torch.float)
        batch_boundary_label_tensor = torch.ones(len(batch_input), max_length, dtype=torch.long) * (-100)
        
        if self.load_fg_bg_feat and batch_fg_bg_input[0] is not None:
            fg_bg_shape = batch_fg_bg_input[0].shape
            batch_fg_bg_input_tensor = torch.zeros(
                len(batch_input), *fg_bg_shape[:-1], max_length, dtype=torch.float
            )
        else:
            batch_fg_bg_input_tensor = torch.zeros(len(batch_input), 1, max_length, dtype=torch.float)
        
        # Fill tensors
        for i in range(len(batch_input)):
            seq_length = batch_input[i].shape[1]
            batch_input_tensor[i, :, :seq_length] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :len(batch_target[i])] = torch.from_numpy(batch_target[i])
            batch_boundary_label_tensor[i, :len(batch_boundary_label[i])] = torch.from_numpy(batch_boundary_label[i])
            mask[i, :, :len(batch_target[i])] = torch.ones(self.num_classes, len(batch_target[i]))
            
            if self.load_fg_bg_feat and batch_fg_bg_input[i] is not None:
                fg_bg_length = batch_fg_bg_input[i].shape[-1]
                batch_fg_bg_input_tensor[i, ..., :fg_bg_length] = torch.from_numpy(batch_fg_bg_input[i])
        
        return {
            'batch_vid': '-'.join(batch_vid),
            'batch_input': batch_input_tensor,
            'batch_target': batch_target_tensor,
            'batch_fg_bg_feat': batch_fg_bg_input_tensor,
            'mask': mask,
            'batch_boundary_label': batch_boundary_label_tensor
        }

    def next_batch(self, batch_size):
        """
        Generate the next batch of data.
        
        Args:
            batch_size (int): Size of the batch
            
        Returns:
            dict: Dictionary containing batch data as tensors
        """
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        batch_fg_bg_input = []
        batch_task_label = []
        batch_vid = []
        
        # Load data for each video in batch
        for vid in batch:
            batch_vid.append(vid)
            
            features, classes, fg_bg_features = self._load_video_data(vid)
            
            # Apply sampling
            batch_input.append(features[:, ::self.sample_rate])
            batch_target.append(classes[::self.sample_rate])
            batch_task_label.append(vid.split('_')[0])
            
            if fg_bg_features is not None:
                batch_fg_bg_input.append(fg_bg_features[:, ::self.sample_rate])
            else:
                batch_fg_bg_input.append(None)
        
        # Apply trimming and concatenation if needed
        has_multitask = any('multitask' in vid for vid in batch)
        
        if self.num_video_concat > 1 and not has_multitask:
            if self.load_fg_bg_feat and all(feat is not None for feat in batch_fg_bg_input):
                batch_input, batch_fg_bg_input, batch_target, batch_boundary_label = \
                    trim_and_concat_with_boundaries(
                        batch_input, batch_fg_bg_input, 
                        batch_target=batch_target, 
                        batch_task_label=batch_task_label, 
                        max_num_point=self.max_num_point
                    )
            else:
                batch_input, batch_target, batch_boundary_label = \
                    trim_and_concat_with_boundaries(
                        batch_input, 
                        batch_target=batch_target, 
                        batch_task_label=batch_task_label, 
                        max_num_point=self.max_num_point
                    )
                batch_fg_bg_input = [[None] for _ in batch_input]
        else:
            # No trimming - create boundary labels as zeros
            batch_boundary_label = [np.zeros_like(target) for target in batch_target]
        
        # Convert to tensors
        return self._create_tensors(
            batch_input, batch_target, batch_fg_bg_input, batch_boundary_label, batch_vid
        )