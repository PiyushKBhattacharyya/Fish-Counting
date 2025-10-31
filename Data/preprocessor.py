import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union
import albumentations as A
from pathlib import Path

try:
    from ..core.config import config
    from ..core.logger import logger
except ImportError:
    from core.config import config
    from core.logger import logger


class SonarPreprocessor:
    """Advanced preprocessor optimized for sonar fish counting data."""

    def __init__(self):
        self.image_size = config.get('data.image_size', [640, 640])
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        # Sonar-specific preprocessing
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.bilateral_filter = cv2.bilateralFilter

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess a single sonar image with sonar-specific enhancements.

        Args:
            image: Input image (H, W, C) or (H, W)

        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Apply CLAHE for contrast enhancement (sonar-specific)
        enhanced = self.clahe.apply(gray.astype(np.uint8))

        # Apply bilateral filter to reduce noise while preserving edges
        filtered = self.bilateral_filter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

        # Convert back to RGB by duplicating channels
        processed = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)

        return processed

    def preprocess_sequence(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocess a sequence of images with temporal consistency.

        Args:
            images: List of input images

        Returns:
            Preprocessed sequence tensor (T, C, H, W)
        """
        processed_images = []

        for img in images:
            processed = self.preprocess_image(img)

            # Resize to target size
            processed = cv2.resize(processed, tuple(self.image_size[::-1]))

            # Normalize
            processed = processed.astype(np.float32) / 255.0
            processed = (processed - self.mean) / self.std

            # Convert to tensor and permute to (C, H, W)
            tensor = torch.from_numpy(processed).permute(2, 0, 1)
            processed_images.append(tensor)

        # Stack into sequence
        sequence = torch.stack(processed_images)

        return sequence

    def augment_sequence(
        self,
        images: torch.Tensor,
        bboxes: List[List],
        temporal_consistency: bool = True
    ) -> Tuple[torch.Tensor, List[List]]:
        """
        Apply augmentations to a sequence while maintaining temporal consistency.

        Args:
            images: Sequence tensor (T, C, H, W)
            bboxes: List of bbox lists for each frame
            temporal_consistency: Whether to apply consistent augmentations across frames

        Returns:
            Augmented images and bboxes
        """
        T, C, H, W = images.shape

        # Convert to numpy for albumentations
        image_list = []
        for i in range(T):
            img_np = images[i].permute(1, 2, 0).numpy()
            img_np = (img_np * self.std + self.mean) * 255.0
            img_np = img_np.astype(np.uint8)
            image_list.append(img_np)

        # Apply temporal-consistent augmentations
        augmented_images = []
        augmented_bboxes = []

        if temporal_consistency:
            # Use same transformation for all frames (except spatial)
            base_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussianBlur(p=0.1),
            ])

            # Apply same transform to all frames
            transformed_first = base_transform(image=image_list[0], bboxes=bboxes[0])
            base_transform_params = transformed_first

            for i in range(T):
                # Apply same spatial transforms
                transformed = A.Compose([
                    A.HorizontalFlip(p=base_transform_params.get('HorizontalFlip', False)),
                    A.VerticalFlip(p=base_transform_params.get('VerticalFlip', False)),
                    A.RandomBrightnessContrast(
                        brightness_limit=base_transform_params.get('RandomBrightnessContrast', {}).get('brightness', 0),
                        contrast_limit=base_transform_params.get('RandomBrightnessContrast', {}).get('contrast', 0),
                        p=1.0
                    ),
                    A.GaussianBlur(
                        blur_limit=base_transform_params.get('GaussianBlur', {}).get('blur_limit', 0),
                        p=base_transform_params.get('GaussianBlur', {}).get('p', 0)
                    ),
                ])(image=image_list[i], bboxes=bboxes[i])

                augmented_images.append(transformed['image'])
                augmented_bboxes.append(transformed['bboxes'])
        else:
            # Apply independent augmentations
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussianBlur(p=0.1),
            ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

            for i in range(T):
                transformed = transform(image=image_list[i], bboxes=bboxes[i])
                augmented_images.append(transformed['image'])
                augmented_bboxes.append(transformed['bboxes'])

        # Convert back to tensors
        processed_images = []
        for img in augmented_images:
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            img_tensor = (img_tensor - torch.from_numpy(self.mean).view(-1, 1, 1)) / torch.from_numpy(self.std).view(-1, 1, 1)
            processed_images.append(img_tensor)

        sequence = torch.stack(processed_images)

        return sequence, augmented_bboxes

    def extract_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract sonar-specific features from the image.

        Args:
            image: Input image

        Returns:
            Dictionary of extracted features
        """
        features = {}

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Edge detection (Canny)
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        features['edges'] = edges

        # Histogram of oriented gradients (simplified)
        # This is a basic implementation - in practice, you'd use skimage.feature.hog
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        features['gradients'] = magnitude.astype(np.uint8)

        # Local binary patterns (simplified)
        # This is a basic implementation - in practice, you'd use skimage.feature.local_binary_pattern
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.uint8)
        eroded = cv2.erode(gray.astype(np.uint8), kernel, iterations=1)
        features['texture'] = eroded

        return features

    def detect_motion_regions(
        self,
        sequence: torch.Tensor,
        threshold: float = 0.1
    ) -> torch.Tensor:
        """
        Detect motion regions in a sequence using frame differences.

        Args:
            sequence: Input sequence (T, C, H, W)
            threshold: Motion detection threshold

        Returns:
            Motion mask (T, 1, H, W)
        """
        T, C, H, W = sequence.shape
        motion_masks = []

        for i in range(1, T):
            # Compute frame difference
            diff = torch.abs(sequence[i] - sequence[i-1]).mean(dim=0, keepdim=True)
            motion_mask = (diff > threshold).float()
            motion_masks.append(motion_mask)

        # Add zero motion for first frame
        motion_masks.insert(0, torch.zeros_like(motion_masks[0]))

        return torch.stack(motion_masks)


class DataPreprocessor:
    """Main data preprocessing pipeline."""

    def __init__(self):
        self.sonar_processor = SonarPreprocessor()

    def __call__(
        self,
        batch: Dict[str, Union[torch.Tensor, List]]
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """
        Process a batch of data.

        Args:
            batch: Input batch with 'images' and 'targets'

        Returns:
            Processed batch
        """
        images = batch['images']  # (B, T, C, H, W)
        targets = batch['targets']  # List of lists of tensors

        processed_batch = {
            'images': images,
            'targets': targets,
            'sequence_ids': batch.get('sequence_ids', []),
            'locations': batch.get('locations', [])
        }

        # Add motion features if configured
        if config.get('model.temporal_attention', True):
            motion_features = []
            for seq in images:
                motion = self.sonar_processor.detect_motion_regions(seq)
                motion_features.append(motion)
            processed_batch['motion_masks'] = torch.stack(motion_features)

        return processed_batch