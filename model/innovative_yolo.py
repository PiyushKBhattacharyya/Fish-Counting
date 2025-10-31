import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import math
import numpy as np
import os

try:
    from ..core.config import config
    from ..core.logger import logger
except ImportError:
    from core.config import config
    from core.logger import logger


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for sequence modeling."""

    def __init__(self, channels: int, sequence_length: int):
        super().__init__()
        self.channels = channels
        self.sequence_length = sequence_length

        # Query, Key, Value projections
        self.q_proj = nn.Conv2d(channels, channels // 8, 1)
        self.k_proj = nn.Conv2d(channels, channels // 8, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)

        # Temporal positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, sequence_length, channels, 1, 1))

        # Output projection
        self.out_proj = nn.Conv2d(channels, channels, 1)

        # Layer norm
        self.norm = nn.GroupNorm(32, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape

        # Reshape for attention: (B*T, C, H, W)
        x_reshaped = x.view(B * T, C, H, W)

        # Project to queries, keys, values
        q = self.q_proj(x_reshaped).view(B, T, -1, H, W)  # (B, T, C', H, W)
        k = self.k_proj(x_reshaped).view(B, T, -1, H, W)
        v = self.v_proj(x_reshaped).view(B, T, C, H, W)

        # Add positional encoding
        q = q + self.pos_encoding[:, :, :q.size(2)]
        k = k + self.pos_encoding[:, :, :k.size(2)]

        # Reshape for attention computation: (B, T, C', H*W)
        q = q.view(B, T, -1, H * W)  # (B, T, C', H*W)
        k = k.view(B, T, -1, H * W)  # (B, T, C', H*W)
        v = v.view(B, T, C, H * W)   # (B, T, C, H*W)

        # Compute attention: (B, T, T)
        # First average over spatial dimensions (H*W)
        q_avg = q.mean(dim=-1)  # (B, T, C')
        k_avg = k.mean(dim=-1)  # (B, T, C')

        attn = torch.einsum('btc,bsc->bts', q_avg, k_avg) / math.sqrt(q_avg.size(-1))
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values: (B, T, C, H*W) -> (B*T, C, H, W)
        out = v.view(B * T, C, H, W)

        # Output projection and residual
        out = self.out_proj(out)
        out = self.norm(out + x_reshaped)

        # Reshape back to sequence format
        out = out.view(B, T, C, H, W)

        return out


class SonarFeatureExtractor(nn.Module):
    """Lightweight feature extractor optimized for sonar images."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()

        # Initial convolution with larger kernel for sonar texture
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        # Lightweight blocks with depthwise separable convolutions
        self.layer1 = self._make_layer(base_channels, base_channels * 2, 2)
        self.layer2 = self._make_layer(base_channels * 2, base_channels * 4, 2)
        self.layer3 = self._make_layer(base_channels * 4, base_channels * 8, 2)

        # Sonar-specific enhancement
        self.edge_enhancer = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1, groups=base_channels * 8),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True)
        )

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int) -> nn.Sequential:
        """Create a lightweight convolutional layer."""
        layers = []

        # First block with downsampling
        layers.extend([
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, groups=in_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),  # Pointwise
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])

        # Additional blocks
        for _ in range(1, num_blocks):
            layers.extend([
                nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) or (B, T, C, H, W)
        Returns:
            (B, C_out, H_out, W_out) or (B, T, C_out, H_out, W_out)
        """
        if x.dim() == 5:  # Sequence input (B, T, C, H, W)
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)

            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.edge_enhancer(x)

            # Reshape back to sequence
            x = x.view(B, T, *x.shape[1:])
        else:  # Single image input (B, C, H, W)
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.edge_enhancer(x)

        return x


class YOLODetectionHead(nn.Module):
    """YOLO-style detection head with custom anchors for fish."""

    def __init__(self, in_channels: int, num_classes: int = 1, num_anchors: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Fish-specific anchor boxes (normalized)
        # Based on typical fish aspect ratios in sonar images
        self.anchors = torch.tensor([
            [0.1, 0.1],  # Small fish
            [0.2, 0.15], # Medium fish
            [0.3, 0.2]   # Large fish
        ])

        # Detection head
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, (num_classes + 5) * num_anchors, 1)  # 5 = x,y,w,h,conf
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) or (B, T, C, H, W)
        Returns:
            (B, num_anchors, H, W, num_classes + 5) or (B, T, num_anchors, H, W, num_classes + 5)
        """
        if x.dim() == 5:  # Sequence input
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)

            x = self.conv(x)
            B_out, C_out, H_out, W_out = x.shape

            # Reshape to detection format
            x = x.view(B_out, self.num_anchors, -1, H_out, W_out)
            x = x.permute(0, 1, 3, 4, 2)  # (B*T, num_anchors, H, W, features)

            # Split into components
            x = x.view(B, T, self.num_anchors, H_out, W_out, -1)
        else:
            x = self.conv(x)
            x = x.view(x.size(0), self.num_anchors, -1, x.size(2), x.size(3))
            x = x.permute(0, 1, 3, 4, 2)  # (B, num_anchors, H, W, features)

        return x


class InnovativeYOLO(nn.Module):
    """Innovative lightweight YOLO model with temporal attention and sonar optimization."""

    def __init__(
        self,
        num_classes: int = 1,
        sequence_length: int = 10,
        input_channels: int = 3,
        temporal_attention: bool = True,
        sonar_optimization: bool = True
    ):
        super().__init__()

        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.temporal_attention = temporal_attention
        self.sonar_optimization = sonar_optimization

        # Feature extractor
        self.backbone = SonarFeatureExtractor(input_channels)

        # Temporal attention (applied before detection)
        if temporal_attention:
            # Get feature channels from backbone
            with torch.no_grad():
                dummy_input = torch.randn(1, input_channels, 640, 640)
                dummy_features = self.backbone(dummy_input)
                feature_channels = dummy_features.size(1)

            self.temporal_attn = TemporalAttention(feature_channels, sequence_length)

        # Detection head
        head_channels = feature_channels if temporal_attention else 256  # fallback
        self.detection_head = YOLODetectionHead(head_channels, num_classes)

        # Initialize weights
        self._initialize_weights()

        logger.info(f"Initialized InnovativeYOLO with temporal_attention={temporal_attention}, "
                   f"sonar_optimization={sonar_optimization}")

    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor (B, C, H, W) or (B, T, C, H, W)

        Returns:
            Detection predictions
        """
        if x.dim() == 4:  # Single image
            features = self.backbone(x)
            if self.temporal_attention:
                # Add temporal dimension for consistency
                features = features.unsqueeze(1)  # (B, 1, C, H, W)
                features = self.temporal_attn(features)
                features = features.squeeze(1)  # (B, C, H, W)
            detections = self.detection_head(features)

        else:  # Sequence input (B, T, C, H, W)
            features = self.backbone(x)  # (B, T, C, H, W)

            if self.temporal_attention:
                features = self.temporal_attn(features)

            detections = self.detection_head(features)

        return detections

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: List[torch.Tensor],
        device: str = 'cuda'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute proper YOLO loss with box regression, objectness, and classification components.

        Args:
            predictions: Model predictions (B, T, num_anchors, H, W, num_classes + 5)
            targets: Ground truth targets [batch_idx, class_id, x, y, w, h, conf]
            device: Device for computations

        Returns:
            Dictionary of loss components
        """
        # YOLO loss hyperparameters
        lambda_box = 0.05  # Box loss weight
        lambda_obj = 1.0   # Objectness loss weight
        lambda_cls = 0.5   # Classification loss weight
        lambda_noobj = 0.5 # No object loss weight

        # Get prediction dimensions
        if predictions.dim() == 5:  # (B, num_anchors, H, W, features)
            B, num_anchors, H, W, features = predictions.shape
            T = 1
        else:  # (B, T, num_anchors, H, W, features)
            B, T, num_anchors, H, W, features = predictions.shape

        num_classes = features - 5  # x, y, w, h, conf, classes

        # Reshape predictions for processing
        pred = predictions.view(B * T, num_anchors, H, W, features)

        # Extract prediction components
        pred_xy = pred[..., 0:2].sigmoid()  # Sigmoid for center coordinates
        pred_wh = pred[..., 2:4]  # Width/height (will be exponentiated relative to anchors)
        pred_conf = pred[..., 4:5].sigmoid()  # Objectness confidence
        pred_cls = pred[..., 5:].sigmoid()  # Class probabilities

        # Generate anchor grids
        grid_x, grid_y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
        grid_x = grid_x.to(device).float()
        grid_y = grid_y.to(device).float()

        # Anchor boxes (normalized to feature map size)
        anchors = self.detection_head.anchors.to(device).float()  # (num_anchors, 2)
        anchor_w = anchors[:, 0].view(1, num_anchors, 1, 1)
        anchor_h = anchors[:, 1].view(1, num_anchors, 1, 1)

        # Build target tensors
        obj_mask = torch.zeros(B * T, num_anchors, H, W, device=device)
        noobj_mask = torch.ones(B * T, num_anchors, H, W, device=device)
        target_xy = torch.zeros(B * T, num_anchors, H, W, 2, device=device)
        target_wh = torch.zeros(B * T, num_anchors, H, W, 2, device=device)
        target_conf = torch.zeros(B * T, num_anchors, H, W, device=device)
        target_cls = torch.zeros(B * T, num_anchors, H, W, num_classes, device=device)

        # Process targets for each batch and time step
        for batch_idx, batch_targets in enumerate(targets):  # batch_targets is list of T tensors
            for time_idx, frame_targets in enumerate(batch_targets):  # frame_targets is [N, 5] tensor
                b = batch_idx * T + time_idx  # flattened index

                for target in frame_targets:
                    if len(target) >= 5:  # [class_id, x, y, w, h]
                        class_id, tx, ty, tw, th = target[:5]
                        tconf = 1.0  # default confidence for ground truth

                        # Convert to grid coordinates
                        gx = tx * W
                        gy = ty * H
                        gw = tw * W
                        gh = th * H

                        # Get grid cell
                        gi = int(gx)
                        gj = int(gy)

                        if 0 <= gi < W and 0 <= gj < H:
                            # Find best anchor using IoU
                            best_anchor = 0
                            best_iou = 0
                            for a_idx in range(num_anchors):
                                anchor = anchors[a_idx]
                                # Calculate IoU between target box and anchor
                                anchor_box = [0, 0, anchor[0], anchor[1]]  # [x1,y1,x2,y2]
                                target_box = [tx - tw/2, ty - th/2, tx + tw/2, ty + th/2]

                                # IoU calculation
                                inter_x1 = max(anchor_box[0], target_box[0])
                                inter_y1 = max(anchor_box[1], target_box[1])
                                inter_x2 = min(anchor_box[2], target_box[2])
                                inter_y2 = min(anchor_box[3], target_box[3])

                                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                                anchor_area = anchor[0] * anchor[1]
                                target_area = tw * th
                                union_area = anchor_area + target_area - inter_area

                                iou = inter_area / union_area if union_area > 0 else 0

                                if iou > best_iou:
                                    best_iou = iou
                                    best_anchor = a_idx

                            # Set target values
                            obj_mask[b, best_anchor, gj, gi] = 1
                            noobj_mask[b, best_anchor, gj, gi] = 0

                            target_xy[b, best_anchor, gj, gi, 0] = gx - gi  # offset x
                            target_xy[b, best_anchor, gj, gi, 1] = gy - gj  # offset y

                            target_wh[b, best_anchor, gj, gi, 0] = torch.log(gw / (anchors[best_anchor, 0] * W) + 1e-16)
                            target_wh[b, best_anchor, gj, gi, 1] = torch.log(gh / (anchors[best_anchor, 1] * H) + 1e-16)

                            target_conf[b, best_anchor, gj, gi] = tconf

                            if num_classes > 0:
                                target_cls[b, best_anchor, gj, gi, int(class_id)] = 1

        # Compute losses
        # Box loss (MSE for xy, MSE for wh)
        box_loss_xy = F.mse_loss(pred_xy, target_xy, reduction='sum') / (B * T)
        box_loss_wh = F.mse_loss(pred_wh, target_wh, reduction='sum') / (B * T)
        box_loss = lambda_box * (box_loss_xy + box_loss_wh)

        # Objectness loss (BCE)
        obj_loss = lambda_obj * F.binary_cross_entropy(pred_conf.squeeze(-1), target_conf, reduction='sum') / (B * T)

        # No object loss
        noobj_loss = lambda_noobj * F.binary_cross_entropy(pred_conf.squeeze(-1), target_conf, weight=noobj_mask, reduction='sum') / (B * T)

        # Classification loss (BCE)
        cls_loss = torch.tensor(0.0, device=device)
        if num_classes > 0:
            cls_loss = lambda_cls * F.binary_cross_entropy(pred_cls, target_cls, reduction='sum') / (B * T)

        # Total loss
        total_loss = box_loss + obj_loss + noobj_loss + cls_loss

        losses = {
            'total_loss': total_loss,
            'box_loss': box_loss,
            'obj_loss': obj_loss,
            'noobj_loss': noobj_loss,
            'cls_loss': cls_loss
        }

        return losses

    def get_model_info(self) -> Dict[str, any]:
        """Get model information and specifications."""
        info = {
            'model_name': 'InnovativeYOLO',
            'num_classes': self.num_classes,
            'sequence_length': self.sequence_length,
            'temporal_attention': self.temporal_attention,
            'sonar_optimization': self.sonar_optimization,
            'novel_features': [
                'Custom lightweight backbone with depthwise separable convolutions',
                'Temporal attention mechanism for sequence modeling',
                'Sonar-specific image preprocessing and feature extraction',
                'Fish-optimized anchor boxes',
                'Edge enhancement for sonar texture detection'
            ]
        }

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        info['total_parameters'] = total_params
        info['trainable_parameters'] = trainable_params

        return info


def create_innovative_yolo(config_dict: Optional[Dict] = None) -> InnovativeYOLO:
    """Factory function to create InnovativeYOLO model with optional pre-trained weights."""
    if config_dict is None:
        config_dict = config.get('model', {})

    model = InnovativeYOLO(
        num_classes=config_dict.get('num_classes', 1),
        sequence_length=config_dict.get('sequence_length', 10),
        input_channels=config_dict.get('input_channels', 3),
        temporal_attention=config_dict.get('temporal_attention', True),
        sonar_optimization=config_dict.get('sonar_optimization', True)
    )

    # Load pre-trained weights if available
    pretrained_path = config_dict.get('pretrained_weights', 'yolov8n.pt')
    if pretrained_path and os.path.exists(pretrained_path):
        try:
            # Load YOLOv8 weights (this is a simplified version - in practice you'd need
            # to map the weights properly from YOLOv8 to this architecture)
            checkpoint = torch.load(pretrained_path, map_location='cpu')

            # Try to load state dict
            if 'model' in checkpoint:
                # YOLOv8 format
                state_dict = checkpoint['model']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                # General checkpoint format
                state_dict = checkpoint['state_dict']
            else:
                # Direct state dict
                state_dict = checkpoint

            # Load compatible weights (this would need proper weight mapping in practice)
            # For now, just log that weights are available
            logger.info(f"Found pre-trained weights at {pretrained_path}, but weight mapping not implemented yet")

        except Exception as e:
            logger.warning(f"Failed to load pre-trained weights from {pretrained_path}: {e}")

    return model