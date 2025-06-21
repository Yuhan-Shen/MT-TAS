#!/usr/bin/env python3
"""
Main script for temporal action segmentation training and inference.

This script provides a unified interface for training and evaluating temporal action
segmentation models with various optional components like foreground-aware refinement,
segment boundary learning, and foreground-background feature composition.
"""

import torch
from model import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import random
from eval import evaluate
import logging
from datetime import datetime


def setup_logging(model_dir, exp_id):
    """Setup logging configuration."""
    logger = logging.getLogger(f'ActionSegmentation_{exp_id}')
    logger.setLevel(logging.INFO)
    
    # Create log filename with timestamp
    current_time = datetime.now()
    log_filename = current_time.strftime("%Y-%m-%d_%H-%M-%S.log")
    
    # File handler
    file_handler = logging.FileHandler(os.path.join(model_dir, log_filename))
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Temporal Action Segmentation')
    
    # Basic settings
    parser.add_argument('--action', default='train', choices=['train', 'predict'],
                       help='Action to perform: train or predict')
    parser.add_argument('--dataset', default='meka')
    parser.add_argument('--split', default='1', 
                       help='Dataset split number')
    parser.add_argument('--exp_id', default='mstcn', type=str,
                       help='Experiment identifier')
    
    # Training parameters
    parser.add_argument('--batch_size', default=1, type=int,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', default=50, type=int,
                       help='Number of training epochs')
    parser.add_argument('--lr', default=0.0005, type=float,
                       help='Learning rate')
    parser.add_argument('--seed', default=1538574472, type=int,
                       help='Random seed')
    
    # Model configuration
    parser.add_argument('--features_dim', default=2048, type=int,
                       help='Input feature dimension')
    parser.add_argument('--sample_rate', default=1, type=int,
                       help='Frame sampling rate')
    parser.add_argument('--prev_model_path', default='', type=str,
                       help='Path to previous model for initialization')
    
    # Model components
    parser.add_argument('--use_sbl', action='store_true',
                       help='Use Segment Boundary Learning')
    parser.add_argument('--use_fbfc', action='store_true',
                       help='Use Foreground-Background Feature Composition')
    parser.add_argument('--no_faar', action='store_true',
                       help='Disable Foreground-Aware Refinement')
    parser.add_argument('--dual_use_att', action='store_true',
                       help='Use attention in dual encoder')
    
    # Data paths
    parser.add_argument('--train_split', default='', type=str,
                       help='Training split (if different from split)')
    parser.add_argument('--test_split', default='', type=str,
                       help='Test split (if different from split)')
    parser.add_argument('--features_path', default='', type=str,
                       help='Path to feature files')
    parser.add_argument('--fg_bg_features_path', default='', type=str,
                       help='Path to foreground/background features')
    
    # Data processing
    parser.add_argument('--num_video_concat', default=1, type=int,
                       help='Number of videos to concatenate')
    parser.add_argument('--max_num_point', default=-1, type=int,
                       help='Maximum number of frames per video')
    
    return parser.parse_args()


def setup_paths_and_config(args):
    """Setup data paths and dataset-specific configurations."""
    # Set default splits
    if not args.train_split:
        args.train_split = args.split
    if not args.test_split:
        args.test_split = args.split
    
    # Setup file paths
    data_dir = f"./data/{args.dataset}"
    vid_list_file = f"{data_dir}/splits/train.split{args.train_split}.bundle"
    vid_list_file_tst = f"{data_dir}/splits/test.split{args.test_split}.bundle"
    
    # Features path
    if not args.features_path:
        features_path = f"{data_dir}/features/"
    else:
        features_path = args.features_path.rstrip('/') + '/'
    
    # Ground truth path
    gt_path = f"{data_dir}/groundTruth/"
    
    # Foreground/background features path
    if not args.fg_bg_features_path and (args.use_sbl or args.use_fbfc or not args.no_faar):
        # Set default paths for different datasets
        fg_bg_features_path = '/path/to/meka/fg_bg_features'  # Update with actual path
    else:
        fg_bg_features_path = args.fg_bg_features_path
    
    # Mapping file
    mapping_file = f"{data_dir}/mapping.txt"
    
    # Output directories
    model_dir = f"./models/{args.exp_id}/{args.dataset}/split_{args.train_split}"
    results_dir = f"./results/{args.exp_id}/{args.dataset}/epoch{args.num_epochs}/split_{args.train_split}"
    
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Dataset-specific configurations
    sample_rate = args.sample_rate
    
    # Dataset-specific settings
    map_delimiter = '|'
    feature_transpose = True
    
    return {
        'vid_list_file': vid_list_file,
        'vid_list_file_tst': vid_list_file_tst,
        'features_path': features_path,
        'fg_bg_features_path': fg_bg_features_path,
        'gt_path': gt_path,
        'mapping_file': mapping_file,
        'model_dir': model_dir,
        'results_dir': results_dir,
        'sample_rate': sample_rate,
        'map_delimiter': map_delimiter,
        'feature_transpose': feature_transpose
    }


def load_actions_dict(mapping_file, map_delimiter):
    """Load actions dictionary from mapping file."""
    with open(mapping_file, 'r') as f:
        actions = f.read().split('\n')[:-1]
    
    actions_dict = {}
    for action in actions:
        parts = action.split(map_delimiter)
        actions_dict[parts[1]] = int(parts[0])
    
    return actions_dict


def set_random_seeds(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup paths and configurations
    config = setup_paths_and_config(args)
    
    # Setup logging
    logger = setup_logging(config['model_dir'], args.exp_id)
    logger.info(f"Arguments: {args}")
    logger.info(f"Using device: {device}")
    
    # Load actions dictionary
    actions_dict = load_actions_dict(config['mapping_file'], config['map_delimiter'])
    num_classes = len(actions_dict)
    logger.info(f"Number of classes: {num_classes}")
    
    # Model configuration
    num_stages = 4
    num_layers = 10
    num_f_maps = 64
    features_dim = args.features_dim
    
    # Process max_num_point
    max_num_point = args.max_num_point if args.max_num_point >= 0 else None
    
    # Create trainer
    trainer = Trainer(
        num_blocks=num_stages,
        num_layers=num_layers,
        num_f_maps=num_f_maps,
        dim=features_dim,
        num_classes=num_classes,
        logger=logger,
        prev_model_path=args.prev_model_path if args.prev_model_path else None,
        dual_use_att=args.dual_use_att,
        use_sbl=args.use_sbl,
        use_fbfc=args.use_fbfc,
        use_faar=not args.no_faar
    )
    
    if args.action == "train":
        logger.info("Starting training...")
        
        # Create training batch generator
        batch_gen = BatchGenerator(
            num_classes=num_classes,
            actions_dict=actions_dict,
            gt_path=config['gt_path'],
            features_path=config['features_path'],
            sample_rate=config['sample_rate'],
            feature_transpose=config['feature_transpose'],
            num_video_concat=args.num_video_concat,
            max_num_point=max_num_point,
            fg_bg_features_path=config['fg_bg_features_path']
        )
        batch_gen.read_data(config['vid_list_file'])
        
        # Train the model
        trainer.train(
            save_dir=config['model_dir'],
            batch_gen=batch_gen,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device
        )
        
        logger.info("Training completed. Starting evaluation...")
        
        # Create test batch generator for prediction
        test_batch_gen = BatchGenerator(
            num_classes=num_classes,
            actions_dict=actions_dict,
            gt_path=config['gt_path'],
            features_path=config['features_path'],
            sample_rate=config['sample_rate'],
            feature_transpose=config['feature_transpose'],
            num_video_concat=1,
            max_num_point=max_num_point,
            fg_bg_features_path=config['fg_bg_features_path'],
            mode='test'
        )
        test_batch_gen.read_data(config['vid_list_file_tst'])
        
        # Generate predictions
        trainer.predict_with_dataloader(
            model_dir=config['model_dir'],
            results_dir=config['results_dir'],
            batch_gen=test_batch_gen,
            vid_list_file=config['vid_list_file_tst'],
            epoch=args.num_epochs,
            actions_dict=actions_dict,
            device=device,
            sample_rate=config['sample_rate'],
            feature_transpose=config['feature_transpose'],
            map_delimiter=config['map_delimiter']
        )
        
    elif args.action == "predict":
        logger.info("Starting prediction...")
        
        # Create test batch generator
        test_batch_gen = BatchGenerator(
            num_classes=num_classes,
            actions_dict=actions_dict,
            gt_path=config['gt_path'],
            features_path=config['features_path'],
            sample_rate=config['sample_rate'],
            feature_transpose=config['feature_transpose'],
            num_video_concat=1,
            max_num_point=max_num_point,
            fg_bg_features_path=config['fg_bg_features_path'],
            mode='test'
        )
        test_batch_gen.read_data(config['vid_list_file_tst'])
        
        # Generate predictions
        trainer.predict_with_dataloader(
            model_dir=config['model_dir'],
            results_dir=config['results_dir'],
            batch_gen=test_batch_gen,
            vid_list_file=config['vid_list_file_tst'],
            epoch=args.num_epochs,
            actions_dict=actions_dict,
            device=device,
            sample_rate=config['sample_rate'],
            feature_transpose=config['feature_transpose'],
            map_delimiter=config['map_delimiter']
        )
    
    # Evaluate results
    logger.info("Evaluating results...")
    try:
        evaluate(
            dataset=args.dataset,
            results_dir=config['results_dir'],
            split=args.test_split,
            exp_id=args.exp_id,
            num_epochs=args.num_epochs
        )
        logger.info("Evaluation completed successfully.")
    except Exception as e:
        logger.warning(f"Evaluation failed: {e}")
    
    logger.info("Process completed.")


if __name__ == '__main__':
    main()
