import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from typing import Dict, List, Optional, Tuple, Any
import time
import os
from pathlib import Path
import numpy as np

try:
    from ..core.config import config
    from ..core.logger import logger
    from ..model.innovative_yolo import create_innovative_yolo
except ImportError:
    from core.config import config
    from core.logger import logger
    from model.innovative_yolo import create_innovative_yolo
from evaluation.metrics import EvaluationMetrics
from Data.preprocessor import DataPreprocessor


class Trainer:
    """Training pipeline for the innovative YOLO model."""

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        device: str = 'auto'
    ):
        # Device configuration
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # Model
        if model is None:
            self.model = create_innovative_yolo()
        else:
            self.model = model

        self.model.to(self.device)

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Preprocessor
        self.preprocessor = DataPreprocessor()

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.get('device.mixed_precision', True) else None

        # Evaluation metrics
        self.evaluator = EvaluationMetrics()

        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.training_history = []

        # Checkpoint directory
        self.checkpoint_dir = Path('checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)

        logger.info("Trainer initialized successfully")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with configured parameters."""
        optimizer_config = config.get('training', {})

        optimizer_type = 'AdamW'  # Default to AdamW
        lr = optimizer_config.get('learning_rate', 0.001)
        weight_decay = optimizer_config.get('weight_decay', 0.0001)
        momentum = optimizer_config.get('momentum', 0.937)

        if optimizer_type == 'SGD':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=True
            )
        elif optimizer_type == 'Adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:  # AdamW
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )

        return optimizer

    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler."""
        training_config = config.get('training', {})
        epochs = training_config.get('epochs', 100)

        scheduler_type = 'cosine'  # Default

        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=lr * 0.01  # Minimum LR
            )
        elif scheduler_type == 'one_cycle':
            scheduler = OneCycleLR(
                self.optimizer,
                max_lr=config.get('training.learning_rate', 0.001),
                epochs=epochs,
                steps_per_epoch=len(self.train_loader) if self.train_loader else 100,
                pct_start=0.3,
                anneal_strategy='cos'
            )
        else:
            scheduler = None

        return scheduler

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {'loss': 0.0}

        if not self.train_loader:
            logger.warning("No training loader provided")
            return epoch_metrics

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            images = batch['images'].to(self.device)
            targets = batch['targets']  # List of tensors, keep on CPU for processing

            # Preprocess batch
            processed_batch = self.preprocessor(batch)
            if 'motion_masks' in processed_batch:
                motion_masks = processed_batch['motion_masks'].to(self.device)
            else:
                motion_masks = None

            # Forward pass with mixed precision
            self.optimizer.zero_grad()

            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss_dict = self.model.compute_loss(outputs, targets, self.device)

                self.scaler.scale(loss_dict['total_loss']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss_dict = self.model.compute_loss(outputs, targets, self.device)
                loss_dict['total_loss'].backward()
                self.optimizer.step()

            # Record losses
            epoch_losses.append(loss_dict['total_loss'].item())

            # Log progress
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss_dict['total_loss'].item():.4f}")

        # Calculate epoch metrics
        epoch_metrics['loss'] = np.mean(epoch_losses)
        epoch_metrics.update({k: v for k, v in loss_dict.items() if k != 'total_loss'})

        return epoch_metrics

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        val_metrics = {}

        if not self.val_loader:
            logger.warning("No validation loader provided")
            return val_metrics

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['images'].to(self.device)
                targets = batch['targets']

                # Forward pass
                outputs = self.model(images)

                # Store for evaluation
                all_predictions.append(outputs.cpu())
                all_targets.extend(targets)

        # Compute evaluation metrics
        if all_predictions and all_targets:
            val_metrics = self.evaluator.compute_metrics(all_predictions, all_targets)

        return val_metrics

    def train(self, num_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """Main training loop."""
        if num_epochs is None:
            num_epochs = config.get('training.epochs', 100)

        logger.info(f"Starting training for {num_epochs} epochs")

        warmup_epochs = config.get('training.warmup_epochs', 3)
        patience = config.get('training.patience', 10)
        best_metric_key = 'mAP'  # Primary metric to monitor

        patience_counter = 0
        start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate_epoch()

            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, OneCycleLR):
                    self.scheduler.step()
                else:
                    self.scheduler.step()

            # Log epoch results
            epoch_time = time.time() - epoch_start_time
            logger.log_epoch_summary(epoch + 1, train_metrics, val_metrics)
            logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")

            # Save training history
            epoch_history = {
                'epoch': epoch + 1,
                'train': train_metrics,
                'val': val_metrics,
                'lr': self.optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time
            }
            self.training_history.append(epoch_history)

            # Save checkpoint
            self.save_checkpoint(epoch_history)

            # Early stopping
            current_metric = val_metrics.get(best_metric_key, 0.0)
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                patience_counter = 0
                # Save best model
                self.save_checkpoint(epoch_history, is_best=True)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping after {epoch + 1} epochs")
                    break

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")

        return self.get_training_history()

    def save_checkpoint(self, epoch_history: Dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch_history['epoch'],
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler': self.scaler.state_dict() if self.scaler else None,
            'history': epoch_history,
            'best_metric': self.best_metric
        }

        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best_model.pth'
        else:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch_history["epoch"]:03d}.pth'

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler and checkpoint.get('scaler'):
            self.scaler.load_state_dict(checkpoint['scaler'])

        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint.get('best_metric', 0.0)

        logger.info(f"Checkpoint loaded from {checkpoint_path}")

    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history as a dictionary."""
        history = {
            'epochs': [],
            'train_loss': [],
            'val_mAP': [],
            'val_precision': [],
            'val_recall': [],
            'learning_rate': []
        }

        for epoch_data in self.training_history:
            history['epochs'].append(epoch_data['epoch'])
            history['train_loss'].append(epoch_data['train']['loss'])
            history['val_mAP'].append(epoch_data['val'].get('mAP', 0.0))
            history['val_precision'].append(epoch_data['val'].get('precision', 0.0))
            history['val_recall'].append(epoch_data['val'].get('recall', 0.0))
            history['learning_rate'].append(epoch_data['lr'])

        return history


def create_trainer(
    model: Optional[nn.Module] = None,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    device: str = 'auto'
) -> Trainer:
    """Factory function to create trainer."""
    return Trainer(model, train_loader, val_loader, device)