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

    def _count_detections(self, predictions, conf_threshold=0.25, nms_iou_threshold=0.45):
        """
        Count total detections after applying confidence thresholding and NMS.

        Args:
            predictions: Model predictions (B, T, num_anchors, H, W, num_classes + 5)
            conf_threshold: Confidence threshold for filtering
            nms_iou_threshold: IoU threshold for NMS

        Returns:
            Total number of detections after post-processing
        """
        try:
            total_count = 0

            # Handle both single image and sequence inputs
            if predictions.dim() == 5:  # (B, num_anchors, H, W, features)
                B, num_anchors, H, W, features = predictions.shape
                T = 1
                pred = predictions.unsqueeze(1)  # Add T dimension
            else:  # (B, T, num_anchors, H, W, features)
                B, T, num_anchors, H, W, features = predictions.shape
                pred = predictions

            num_classes = features - 5

            # Process each batch and time step
            for b in range(B):
                for t in range(T):
                    batch_pred = pred[b, t]  # (num_anchors, H, W, features)

                    # Extract predictions
                    pred_xy = batch_pred[..., 0:2].sigmoid()
                    pred_wh = batch_pred[..., 2:4]
                    pred_conf = batch_pred[..., 4].sigmoid()
                    pred_cls = batch_pred[..., 5:].sigmoid()

                    # Get anchors
                    anchors = self.model.detection_head.anchors.to(predictions.device)

                    # Convert to absolute coordinates
                    grid_x, grid_y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
                    grid_x = grid_x.to(predictions.device)
                    grid_y = grid_y.to(predictions.device)

                    # Denormalize predictions
                    pred_x = (pred_xy[..., 0] + grid_x) / W
                    pred_y = (pred_xy[..., 1] + grid_y) / H
                    pred_w = torch.exp(pred_wh[..., 0]) * anchors[:, 0].view(num_anchors, 1, 1) / W
                    pred_h = torch.exp(pred_wh[..., 1]) * anchors[:, 1].view(num_anchors, 1, 1) / H

                    # Flatten predictions
                    pred_x = pred_x.view(-1)
                    pred_y = pred_y.view(-1)
                    pred_w = pred_w.view(-1)
                    pred_h = pred_h.view(-1)
                    pred_conf = pred_conf.view(-1)
                    pred_cls = pred_cls.view(num_anchors * H * W, num_classes)

                    # Convert to corner format for NMS
                    pred_x1 = pred_x - pred_w / 2
                    pred_y1 = pred_y - pred_h / 2
                    pred_x2 = pred_x + pred_w / 2
                    pred_y2 = pred_y + pred_h / 2

                    # Filter by confidence
                    conf_mask = pred_conf > conf_threshold
                    if conf_mask.sum() == 0:
                        continue

                    pred_x1 = pred_x1[conf_mask]
                    pred_y1 = pred_y1[conf_mask]
                    pred_x2 = pred_x2[conf_mask]
                    pred_y2 = pred_y2[conf_mask]
                    pred_conf = pred_conf[conf_mask]
                    pred_cls = pred_cls[conf_mask]

                    # Get class predictions
                    if num_classes > 0:
                        class_scores, class_ids = pred_cls.max(dim=1)
                        pred_conf = pred_conf * class_scores  # Multiply by class confidence
                    else:
                        class_ids = torch.zeros_like(pred_conf, dtype=torch.long)

                    # Apply NMS per class
                    if len(pred_x1) > 0:
                        boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)

                        # Simple greedy NMS implementation
                        keep = []
                        scores = pred_conf
                        order = scores.argsort(descending=True)

                        while order.numel() > 0:
                            i = order[0]
                            keep.append(i)

                            if order.numel() == 1:
                                break

                            # Compute IoU with remaining boxes
                            xx1 = torch.max(boxes[i, 0], boxes[order[1:], 0])
                            yy1 = torch.max(boxes[i, 1], boxes[order[1:], 1])
                            xx2 = torch.min(boxes[i, 2], boxes[order[1:], 2])
                            yy2 = torch.min(boxes[i, 3], boxes[order[1:], 3])

                            w = torch.clamp(xx2 - xx1, min=0)
                            h = torch.clamp(yy2 - yy1, min=0)
                            inter = w * h
                            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
                            area_order = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
                            union = area_i + area_order - inter
                            iou = inter / (union + 1e-16)

                            # Keep boxes with IoU < threshold
                            mask = iou <= nms_iou_threshold
                            order = order[1:][mask]

                        total_count += len(keep)

            return total_count
        except Exception as e:
            logger.warning(f"Error in _count_detections: {str(e)}")
            return 0

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
        lr = training_config.get('learning_rate', 0.001)

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

            # Count fish in predictions and ground truth for logging
            with torch.no_grad():
                pred_count = self._count_detections(outputs)
                # Count ground truth detections - handle nested structure
                gt_count = 0
                if targets:
                    for target_sequence in targets:
                        if isinstance(target_sequence, list):
                            for frame_targets in target_sequence:
                                if isinstance(frame_targets, torch.Tensor) and frame_targets.numel() > 0:
                                    gt_count += len(frame_targets)
                        elif isinstance(target_sequence, torch.Tensor) and target_sequence.numel() > 0:
                            gt_count += len(target_sequence)

            # Log progress with detailed metrics
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}/{len(self.train_loader)}, "
                           f"Loss: {loss_dict['total_loss'].item():.4f}, "
                           f"Box: {loss_dict.get('box_loss', 0):.4f}, "
                           f"Obj: {loss_dict.get('obj_loss', 0):.4f}, "
                           f"Cls: {loss_dict.get('cls_loss', 0):.4f}, "
                           f"Pred Fish: {pred_count}, GT Fish: {gt_count}")

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
            for batch_idx, batch in enumerate(self.val_loader):
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
            'val_count_error': [],
            'val_sequence_accuracy': [],
            'learning_rate': []
        }

        for epoch_data in self.training_history:
            history['epochs'].append(epoch_data['epoch'])
            history['train_loss'].append(epoch_data['train']['loss'])
            history['val_mAP'].append(epoch_data['val'].get('mAP', 0.0))
            history['val_precision'].append(epoch_data['val'].get('precision', 0.0))
            history['val_recall'].append(epoch_data['val'].get('recall', 0.0))
            history['val_count_error'].append(epoch_data['val'].get('count_error', 0.0))
            history['val_sequence_accuracy'].append(epoch_data['val'].get('sequence_accuracy', 0.0))
            history['learning_rate'].append(epoch_data['lr'])

        return history

    def plot_training_curves(self, save_path: str = 'training_curves.png'):
        """Plot training curves using matplotlib."""
        try:
            import matplotlib.pyplot as plt

            history = self.get_training_history()

            if not history['epochs']:
                logger.warning("No training history available for plotting")
                return

            epochs = history['epochs']

            # Create individual plots instead of subplots
            base_path = save_path.rsplit('.', 1)[0]  # Remove extension

            # Plot 1: Training Loss
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, history['train_loss'], 'b-', linewidth=2)
            plt.title('Training Loss', fontsize=16)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{base_path}_loss.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Training loss plot saved to {base_path}_loss.png")

            # Plot 2: Validation Detection Metrics
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, history['val_mAP'], 'r-', label='mAP', linewidth=2)
            plt.plot(epochs, history['val_precision'], 'g--', label='Precision', linewidth=2)
            plt.plot(epochs, history['val_recall'], 'm:', label='Recall', linewidth=2)
            plt.title('Validation Detection Metrics', fontsize=16)
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(f"{base_path}_detection_metrics.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Detection metrics plot saved to {base_path}_detection_metrics.png")

            # Plot 3: Count Metrics
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, history['val_count_error'], 'orange', linewidth=2, label='Count Error')
            plt.title('Validation Count Error', fontsize=16)
            plt.xlabel('Epoch')
            plt.ylabel('Absolute Error')
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{base_path}_count_error.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Count error plot saved to {base_path}_count_error.png")

            # Plot 4: Sequence Accuracy and Learning Rate
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(epochs, history['val_sequence_accuracy'], 'purple', linewidth=2, label='Sequence Accuracy')
            ax1.set_title('Sequence Accuracy & Learning Rate', fontsize=16)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy', color='purple')
            ax1.set_ylim(0, 1)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='y', labelcolor='purple')

            # Add learning rate on secondary y-axis
            ax2 = ax1.twinx()
            ax2.plot(epochs, history['learning_rate'], 'gray', linestyle='--', linewidth=1, label='Learning Rate')
            ax2.set_ylabel('Learning Rate', color='gray')
            ax2.tick_params(axis='y', labelcolor='gray')

            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

            plt.savefig(f"{base_path}_sequence_accuracy_lr.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Sequence accuracy and LR plot saved to {base_path}_sequence_accuracy_lr.png")

        except ImportError:
            logger.warning("matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting training curves: {str(e)}")


def create_trainer(
    model: Optional[nn.Module] = None,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    device: str = 'auto'
) -> Trainer:
    """Factory function to create trainer."""
    return Trainer(model, train_loader, val_loader, device)