#!/usr/bin/env python3
"""
Main training script for the Innovative YOLO model on Fish Counting dataset.

This script integrates all submodules:
- Data loading with MOT format annotations
- Innovative YOLO model with temporal attention and sonar optimization
- Training pipeline with custom trainer
- Evaluation metrics for detection and counting accuracy

Note: Designed to run in 'fish_counting_env' virtual environment

Usage:
    python train_innovative_yolo.py --data data.yaml --epochs 100 --batch-size 16
"""
#!/usr/bin/env python3
"""
Main training script for the Innovative YOLO model on Fish Counting dataset.

This script integrates all submodules:
- Data loading with MOT format annotations
- Innovative YOLO model with temporal attention and sonar optimization
- Training pipeline with custom trainer
- Evaluation metrics for detection and counting accuracy

Usage:
    python train_innovative_yolo.py --data data.yaml --epochs 100 --batch-size 16
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import yaml

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.config import config
from core.logger import logger
from Data.loader import FishDataLoader
from model.innovative_yolo import create_innovative_yolo
from training.trainer import create_trainer
from evaluation.metrics import EvaluationMetrics


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Innovative YOLO on Fish Counting Dataset')

    # Data arguments
    parser.add_argument('--data', type=str, default='data.yaml',
                       help='Path to data configuration file')
    parser.add_argument('--data-root', type=str, default='Data/tiny dataset/raw',
                       help='Root directory for dataset')

    # Model arguments
    parser.add_argument('--weights', type=str, default='',
                       help='Initial weights path')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640],
                       help='Image size [height, width]')
    parser.add_argument('--sequence-length', type=int, default=10,
                       help='Sequence length for temporal processing')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, cuda:0, etc.)')

    # Optimization arguments
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                       help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.937,
                       help='SGD momentum')

    # Model features
    parser.add_argument('--temporal-attention', action='store_true', default=True,
                       help='Enable temporal attention mechanism')
    parser.add_argument('--sonar-optimization', action='store_true', default=True,
                       help='Enable sonar-specific optimizations')

    # Evaluation arguments
    parser.add_argument('--conf-thres', type=float, default=0.25,
                       help='Confidence threshold for evaluation')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                       help='IoU threshold for NMS')

    # Logging and saving
    parser.add_argument('--name', type=str, default='innovative_yolo_fish',
                       help='Experiment name')
    parser.add_argument('--project', type=str, default='runs/train',
                       help='Project directory')
    parser.add_argument('--save-period', type=int, default=10,
                       help='Save checkpoint every N epochs')

    return parser.parse_args()


def setup_environment():
    """Setup training environment and verify dependencies."""
    logger.info("Setting up training environment...")

    # Verify we're in the correct virtual environment
    if not sys.executable.endswith('fish_counting_env'):
        logger.warning("Not running in fish_counting_env virtual environment")
        logger.warning(f"Current Python: {sys.executable}")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create necessary directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    logger.info("Environment setup complete")


def load_data_config(data_yaml_path: str) -> dict:
    """Load data configuration from YAML file."""
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    return data_config


def create_data_loaders(args, data_config):
    """Create training, validation, and test data loaders."""
    logger.info("Creating data loaders with specified splits")

    # Get paths from data config
    image_root = data_config.get('image_root', 'Data/tiny dataset/raw')
    annotation_root = data_config.get('annotation_root', 'yolo data')

    # Define location splits
    train_locations = data_config.get('train_locations', ['kenai-train'])
    val_locations = data_config.get('val_locations', ['kenai-val'])
    test_locations = data_config.get('test_locations', ['elwha', 'kenai-channel', 'kenai-rightbank', 'nushagak'])

    logger.info(f"Using roots:")
    logger.info(f"  Images: {image_root}")
    logger.info(f"  Annotations: {annotation_root}")
    logger.info(f"Train locations: {train_locations}")
    logger.info(f"Val locations: {val_locations}")
    logger.info(f"Test locations: {test_locations}")

    # Create data loaders using custom FishDataLoader
    train_loader = FishDataLoader.create_dataloader(
        image_root=image_root,
        annotation_root=annotation_root,
        split='train',
        locations=train_locations,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length
    )

    val_loader = FishDataLoader.create_dataloader(
        image_root=image_root,
        annotation_root=annotation_root,
        split='val',
        locations=val_locations,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length
    )

    test_loader = FishDataLoader.create_dataloader(
        image_root=image_root,
        annotation_root=annotation_root,
        split='test',
        locations=test_locations,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length
    )

    logger.info(f"Data loaders created:")
    logger.info(f"  Train: {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_loader)} batches")
    logger.info(f"  Test: {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


def create_model(args):
    """Create the Innovative YOLO model."""
    logger.info("Creating Innovative YOLO model...")

    # Update config with command line arguments
    config.set('model.temporal_attention', args.temporal_attention)
    config.set('model.sonar_optimization', args.sonar_optimization)
    config.set('model.sequence_length', args.sequence_length)

    model = create_innovative_yolo()

    # Load pretrained weights if specified
    if args.weights and os.path.exists(args.weights):
        logger.info(f"Loading weights from {args.weights}")
        checkpoint = torch.load(args.weights, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    elif args.weights:
        logger.warning(f"Weights file not found: {args.weights}")

    # Get model info
    model_info = model.get_model_info()
    logger.info(f"Model created: {model_info['model_name']}")
    logger.info(f"Parameters: {model_info['total_parameters']:,}")
    logger.info(f"Features: {', '.join(model_info['novel_features'])}")

    return model


def create_optimizer_scheduler(model, args):
    """Create optimizer and learning rate scheduler."""
    # Update training config
    config.set('training.learning_rate', args.lr)
    config.set('training.weight_decay', args.weight_decay)
    config.set('training.momentum', args.momentum)
    config.set('training.epochs', args.epochs)

    # Note: Optimizer and scheduler are created inside the Trainer class
    # We just need to update the config

    return None, None  # Handled by Trainer


def train_model(model, train_loader, val_loader, args):
    """Train the model."""
    logger.info("Starting training...")

    # Create trainer
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device
    )

    # Load checkpoint if resuming
    if args.resume:
        checkpoint_path = f"checkpoints/{args.name}_latest.pth"
        if os.path.exists(checkpoint_path):
            trainer.load_checkpoint(checkpoint_path)
            logger.info(f"Resumed training from {checkpoint_path}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")

    # Train the model
    training_history = trainer.train(num_epochs=args.epochs)

    logger.info("Training completed!")

    return training_history


def evaluate_model(model, test_loader, args):
    """Evaluate the trained model."""
    logger.info("Evaluating model on test set...")

    evaluator = EvaluationMetrics()

    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['images'].to(model.device)
            targets = batch['targets']

            # Forward pass
            outputs = model(images)

            all_predictions.append(outputs.cpu())
            all_targets.extend(targets)

    # Compute metrics
    metrics = evaluator.compute_metrics(all_predictions, all_targets, conf_threshold=args.conf_thres)

    logger.info("Evaluation Results:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")

    # Save results
    evaluator.save_results(f"results/{args.name}_evaluation.json")

    return metrics


def main():
    """Main training function."""
    args = parse_arguments()

    # Setup
    setup_environment()

    # Load data configuration
    data_config = load_data_config(args.data)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(args, data_config)

    # Create model
    model = create_model(args)

    # Train model
    training_history = train_model(model, train_loader, val_loader, args)

    # Evaluate model
    test_metrics = evaluate_model(model, test_loader, args)

    # Save final results
    final_results = {
        'experiment_name': args.name,
        'config': vars(args),
        'training_history': training_history,
        'test_metrics': test_metrics,
        'model_info': model.get_model_info()
    }

    import json
    with open(f"results/{args.name}_final_results.json", 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    logger.info(f"Training pipeline completed. Results saved to results/{args.name}_final_results.json")


if __name__ == '__main__':
    main()