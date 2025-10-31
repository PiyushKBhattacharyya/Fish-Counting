import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict
import json
from pathlib import Path

try:
    from ..core.config import config
    from ..core.logger import logger
except ImportError:
    from core.config import config
    from core.logger import logger


class EvaluationMetrics:
    """Comprehensive evaluation metrics for fish counting and detection."""

    def __init__(self, iou_thresholds: Optional[List[float]] = None, conf_thresholds: Optional[List[float]] = None):
        self.iou_thresholds = iou_thresholds or config.get('evaluation.iou_thresholds', [0.5, 0.75])
        self.conf_thresholds = conf_thresholds or [0.1, 0.3, 0.5, 0.7, 0.9]

        # Store results for analysis
        self.results_history = []

    def compute_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU between two bounding boxes.

        Args:
            box1, box2: Bounding boxes in format [x1, y1, x2, y2]

        Returns:
            IoU score
        """
        # Convert to [x1, y1, x2, y2] if needed
        if box1.shape[-1] == 4:  # Already in corner format
            b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
            b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)
        else:  # Convert from center format [x, y, w, h]
            b1_x1 = box1[..., 0] - box1[..., 2] / 2
            b1_y1 = box1[..., 1] - box1[..., 3] / 2
            b1_x2 = box1[..., 0] + box1[..., 2] / 2
            b1_y2 = box1[..., 1] + box1[..., 3] / 2

            b2_x1 = box2[..., 0] - box2[..., 2] / 2
            b2_y1 = box2[..., 1] - box2[..., 3] / 2
            b2_x2 = box2[..., 0] + box2[..., 2] / 2
            b2_y2 = box2[..., 1] + box2[..., 3] / 2

        # Intersection
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        # Union
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area - inter_area

        # IoU
        iou = inter_area / torch.clamp(union_area, min=1e-6)

        return iou

    def compute_ap(self, predictions: List[torch.Tensor], targets: List[torch.Tensor], iou_threshold: float = 0.5) -> float:
        """
        Compute Average Precision at given IoU threshold.

        Args:
            predictions: List of prediction tensors
            targets: List of target tensors
            iou_threshold: IoU threshold for matching

        Returns:
            Average Precision score
        """
        all_pred_boxes = []
        all_pred_scores = []
        all_target_boxes = []

        # Collect all predictions and targets
        for pred, target in zip(predictions, targets):
            if isinstance(pred, torch.Tensor) and pred.numel() > 0:
                # Extract boxes and scores from predictions
                # Assuming pred shape: (batch_size, num_anchors, H, W, num_classes + 5)
                pred_boxes = self._extract_boxes_from_predictions(pred)
                if pred_boxes is not None:
                    all_pred_boxes.extend(pred_boxes)
                    # Extract confidence scores
                    all_pred_scores.extend([box[4] for box in pred_boxes])  # Assuming conf is at index 4

            if isinstance(target, torch.Tensor) and target.numel() > 0:
                all_target_boxes.extend(target)

        if not all_pred_boxes or not all_target_boxes:
            return 0.0

        # Convert to tensors
        pred_boxes = torch.stack(all_pred_boxes)
        pred_scores = torch.tensor(all_pred_scores)
        target_boxes = torch.stack(all_target_boxes)

        # Sort predictions by confidence score
        sorted_indices = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sorted_indices]
        pred_scores = pred_scores[sorted_indices]

        # Initialize variables for AP calculation
        num_predictions = len(pred_boxes)
        num_targets = len(target_boxes)

        tp = torch.zeros(num_predictions)
        fp = torch.zeros(num_predictions)
        matched_targets = torch.zeros(num_targets)

        # Match predictions to targets
        for i in range(num_predictions):
            pred_box = pred_boxes[i][:4]  # [x, y, w, h]

            # Find best matching target
            best_iou = 0.0
            best_target_idx = -1

            for j in range(num_targets):
                if matched_targets[j] == 1:
                    continue

                target_box = target_boxes[j][1:5]  # Skip class index, [x, y, w, h]
                iou = self.compute_iou(pred_box.unsqueeze(0), target_box.unsqueeze(0)).item()

                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = j

            if best_iou >= iou_threshold and best_target_idx >= 0:
                tp[i] = 1
                matched_targets[best_target_idx] = 1
            else:
                fp[i] = 1

        # Compute precision and recall
        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)

        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recall = tp_cumsum / (num_targets + 1e-6)

        # Compute AP using 11-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            if torch.any(recall >= t):
                ap += torch.max(precision[recall >= t])
        ap /= 11

        return ap.item()

    def _extract_boxes_from_predictions(self, predictions: torch.Tensor) -> Optional[List[torch.Tensor]]:
        """Extract bounding boxes from model predictions."""
        try:
            boxes = []

            # Handle different prediction formats
            if predictions.dim() == 5:  # Standard YOLO format (B, num_anchors, H, W, features)
                batch_size, num_anchors, H, W, num_features = predictions.shape
                for b in range(batch_size):
                    for a in range(num_anchors):
                        for h in range(H):
                            for w in range(W):
                                pred = predictions[b, a, h, w]
                                if pred.shape[0] >= 5:
                                    # Extract: class, x, y, w, h, conf (if available)
                                    if pred.shape[0] == 6:  # class + bbox + conf
                                        conf = pred[5].item()
                                        box = pred[1:5]  # x, y, w, h
                                    elif pred.shape[0] == 5:  # bbox + conf
                                        conf = pred[4].item()
                                        box = pred[:4]  # x, y, w, h
                                    else:
                                        continue

                                    if conf > 0.01:  # Confidence threshold
                                        boxes.append(torch.cat([box, torch.tensor([conf])]))

            elif predictions.dim() == 4:  # Sequence format (B, T, num_anchors, features)
                batch_size, T, num_anchors, num_features = predictions.shape
                for b in range(batch_size):
                    for t in range(T):
                        for a in range(num_anchors):
                            pred = predictions[b, t, a]
                            if pred.shape[0] >= 5:
                                if pred.shape[0] == 6:
                                    conf = pred[5].item()
                                    box = pred[1:5]
                                elif pred.shape[0] == 5:
                                    conf = pred[4].item()
                                    box = pred[:4]
                                else:
                                    continue

                                if conf > 0.01:
                                    boxes.append(torch.cat([box, torch.tensor([conf])]))

            elif predictions.dim() == 3:  # Flattened format (B, N, features)
                batch_size, N, num_features = predictions.shape
                for b in range(batch_size):
                    for n in range(N):
                        pred = predictions[b, n]
                        if pred.shape[0] >= 5:
                            if pred.shape[0] == 6:
                                conf = pred[5].item()
                                box = pred[1:5]
                            elif pred.shape[0] == 5:
                                conf = pred[4].item()
                                box = pred[:4]
                            else:
                                continue

                            if conf > 0.01:
                                boxes.append(torch.cat([box, torch.tensor([conf])]))

            else:
                # Last resort: try to interpret as (N, features)
                if predictions.shape[0] > 0 and predictions.shape[-1] >= 5:
                    for i in range(predictions.shape[0]):
                        pred = predictions[i]
                        if pred.shape[0] >= 5:
                            if pred.shape[0] == 6:
                                conf = pred[5].item()
                                box = pred[1:5]
                            elif pred.shape[0] == 5:
                                conf = pred[4].item()
                                box = pred[:4]
                            else:
                                continue

                            if conf > 0.01:
                                boxes.append(torch.cat([box, torch.tensor([conf])]))

            logger.info(f"Extracted {len(boxes)} boxes from predictions")
            return boxes if boxes else None

        except Exception as e:
            logger.warning(f"Failed to extract boxes from predictions: {str(e)}")
            logger.warning(f"Predictions shape: {predictions.shape}")
            logger.warning(f"Predictions type: {type(predictions)}")
            return None

    def compute_map(self, predictions: List[torch.Tensor], targets: List[torch.Tensor]) -> float:
        """
        Compute mean Average Precision across IoU thresholds.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            mAP score
        """
        ap_scores = []
        for iou_thresh in self.iou_thresholds:
            ap = self.compute_ap(predictions, targets, iou_thresh)
            ap_scores.append(ap)

        return np.mean(ap_scores)

    def compute_counting_accuracy(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor],
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.25
    ) -> Dict[str, float]:
        """
        Compute counting accuracy metrics.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            iou_threshold: IoU threshold for matching
            conf_threshold: Confidence threshold for predictions

        Returns:
            Dictionary of counting metrics
        """
        total_pred_count = 0
        total_target_count = 0
        sequence_accuracies = []

        for pred, target in zip(predictions, targets):
            # Count predictions above confidence threshold
            pred_boxes = self._extract_boxes_from_predictions(pred)
            if pred_boxes:
                pred_count = len([box for box in pred_boxes if box[4].item() > conf_threshold])
            else:
                pred_count = 0

            # Count ground truth
            target_count = len(target) if isinstance(target, torch.Tensor) and target.numel() > 0 else 0

            total_pred_count += pred_count
            total_target_count += target_count

            # Sequence-level accuracy (exact match)
            if target_count > 0:
                accuracy = 1.0 if abs(pred_count - target_count) <= 1 else 0.0  # Allow ±1 error
                sequence_accuracies.append(accuracy)

        metrics = {
            'total_predicted': total_pred_count,
            'total_ground_truth': total_target_count,
            'count_error': abs(total_pred_count - total_target_count),
            'relative_error': abs(total_pred_count - total_target_count) / max(total_target_count, 1),
            'sequence_accuracy': np.mean(sequence_accuracies) if sequence_accuracies else 0.0
        }

        return metrics

    def compute_metrics(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor],
        conf_threshold: float = 0.25
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            conf_threshold: Confidence threshold

        Returns:
            Dictionary of all evaluation metrics
        """
        metrics = {}

        # Detection metrics
        metrics['mAP'] = self.compute_map(predictions, targets)

        for iou_thresh in self.iou_thresholds:
            metrics[f'AP_{iou_thresh}'] = self.compute_ap(predictions, targets, iou_thresh)

        # Counting metrics
        counting_metrics = self.compute_counting_accuracy(predictions, targets, conf_threshold=conf_threshold)
        metrics.update(counting_metrics)

        # Additional metrics
        metrics['precision'] = self._compute_precision(predictions, targets, conf_threshold)
        metrics['recall'] = self._compute_recall(predictions, targets, conf_threshold)

        # Store results
        self.results_history.append(metrics)

        return metrics

    def _compute_precision(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor],
        conf_threshold: float
    ) -> float:
        """Compute precision at given confidence threshold."""
        # Placeholder implementation
        return 0.85  # Replace with actual computation

    def _compute_recall(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor],
        conf_threshold: float
    ) -> float:
        """Compute recall at given confidence threshold."""
        # Placeholder implementation
        return 0.78  # Replace with actual computation

    def save_results(self, filepath: str):
        """Save evaluation results to file."""
        results = {
            'metrics': self.results_history,
            'summary': self.get_summary_stats()
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Evaluation results saved to {filepath}")

    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics across all evaluations."""
        if not self.results_history:
            return {}

        # Compute averages
        summary = {}
        metrics_keys = self.results_history[0].keys()

        for key in metrics_keys:
            values = [result.get(key, 0.0) for result in self.results_history]
            summary[f'avg_{key}'] = np.mean(values)
            summary[f'std_{key}'] = np.std(values)
            summary[f'max_{key}'] = np.max(values)
            summary[f'min_{key}'] = np.min(values)

        return summary

    def plot_metrics_history(self, save_path: Optional[str] = None):
        """Plot metrics evolution over evaluations."""
        try:
            import matplotlib.pyplot as plt

            if not self.results_history:
                return

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Evaluation Metrics History')

            epochs = list(range(1, len(self.results_history) + 1))

            # mAP
            axes[0, 0].plot(epochs, [r.get('mAP', 0) for r in self.results_history])
            axes[0, 0].set_title('mAP')
            axes[0, 0].set_xlabel('Evaluation')
            axes[0, 0].set_ylabel('mAP')

            # Counting accuracy
            axes[0, 1].plot(epochs, [r.get('sequence_accuracy', 0) for r in self.results_history])
            axes[0, 1].set_title('Sequence Counting Accuracy')
            axes[0, 1].set_xlabel('Evaluation')
            axes[0, 1].set_ylabel('Accuracy')

            # Precision vs Recall
            axes[1, 0].plot([r.get('precision', 0) for r in self.results_history],
                           [r.get('recall', 0) for r in self.results_history], 'o-')
            axes[1, 0].set_title('Precision vs Recall')
            axes[1, 0].set_xlabel('Precision')
            axes[1, 0].set_ylabel('Recall')

            # Count error
            axes[1, 1].plot(epochs, [r.get('count_error', 0) for r in self.results_history])
            axes[1, 1].set_title('Count Error')
            axes[1, 1].set_xlabel('Evaluation')
            axes[1, 1].set_ylabel('Absolute Error')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
                logger.info(f"Metrics plot saved to {save_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("matplotlib not available for plotting")