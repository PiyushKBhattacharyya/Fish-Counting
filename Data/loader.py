import os
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Union
import albumentations as A
from PIL import Image

try:
    from ..core.config import config
    from ..core.logger import logger
except ImportError:
    from core.config import config
    from core.logger import logger


class FishDataset(Dataset):
    """Dataset class for fish counting with sequence support."""

    def __init__(
        self,
        image_root: str,
        annotation_root: str,
        split: str = 'train',
        locations: Optional[List[str]] = None,
        transform: Optional[A.Compose] = None,
        sequence_length: int = 10,
        stride: int = 1,
        fish_class_id: int = 0  # Make class ID configurable
    ):
        """
        Initialize the fish dataset.

        Args:
            image_root: Root directory containing image location folders
            annotation_root: Root directory containing annotation location folders
            split: Data split ('train', 'val', 'test')
            locations: List of location names to include for this split
            transform: Albumentations transforms
            sequence_length: Number of frames per sequence
            stride: Stride for sequence extraction
        """
        self.image_root = Path(image_root)
        self.annotation_root = Path(annotation_root)
        self.split = split
        self.locations = locations or []
        self.transform = transform
        self.sequence_length = sequence_length
        self.stride = stride
        self.fish_class_id = fish_class_id

        # Load data annotations
        self.annotations = self._load_annotations()

        # Create sequences
        self.sequences = self._create_sequences()

        logger.info(f"Loaded {len(self.sequences)} sequences for {split} split")

    def _load_annotations(self) -> Dict[str, List[Dict]]:
        """Load annotations from YOLO format files."""
        annotations = {}

        # Filter locations based on specified split locations
        if self.locations:
            locations_to_load = self.locations
        else:
            # Get all location directories from annotation root
            locations_to_load = [d.name for d in self.annotation_root.iterdir() if d.is_dir()]

        for location_name in locations_to_load:
            annotations[location_name] = []

            # Find annotation directory for this location
            annotation_dir = self.annotation_root / location_name
            if not annotation_dir.exists():
                logger.warning(f"Annotation directory not found: {annotation_dir}")
                continue

            # Find all subdirectories (sequences) in annotation location
            sequence_dirs = [d for d in annotation_dir.iterdir() if d.is_dir()]

            for seq_dir in sequence_dirs:
                seq_annotations = self._parse_yolo_sequence(seq_dir, location_name)
                if seq_annotations:
                    annotations[location_name].extend(seq_annotations)

        return annotations

    def _parse_yolo_sequence(self, seq_dir: Path, location_name: str) -> List[Dict]:
        """Parse YOLO format annotation sequence."""
        annotations = []
        sequence_id = seq_dir.name

        try:
            # Get all .txt files in the sequence directory
            txt_files = list(seq_dir.glob('*.txt'))

            for txt_file in txt_files:
                frame_idx = int(txt_file.stem)

                # Read YOLO annotations
                with open(txt_file, 'r') as f:
                    lines = f.readlines()

                bboxes = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if class_id == self.fish_class_id:  # Use configurable class ID
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            conf = float(parts[5]) if len(parts) > 5 else 1.0

                            bboxes.append({
                                'x': x_center,
                                'y': y_center,
                                'w': width,
                                'h': height,
                                'conf': conf
                            })

                # Construct image path from image_root
                image_path = self.image_root / location_name / sequence_id / f"{frame_idx}.jpg"

                if bboxes:  # Only add if there are annotations
                    annotations.append({
                        'frame': frame_idx,
                        'image_path': str(image_path),
                        'bboxes': bboxes,
                        'sequence_id': sequence_id,
                        'location': location_name
                    })

        except Exception as e:
            logger.warning(f"Failed to parse sequence {seq_dir}: {e}")

        return annotations

    def _create_sequences(self) -> List[Dict]:
        """Create sequences from individual frame annotations."""
        sequences = []
        skipped_sequences = 0

        for location, loc_annotations in self.annotations.items():
            # Group by sequence
            sequences_by_id = {}
            for ann in loc_annotations:
                seq_id = ann['sequence_id']
                if seq_id not in sequences_by_id:
                    sequences_by_id[seq_id] = []
                sequences_by_id[seq_id].append(ann)

            # Sort frames and create sequences
            for seq_id, seq_frames in sequences_by_id.items():
                seq_frames.sort(key=lambda x: x['frame'])

                # Create sliding window sequences
                for i in range(0, len(seq_frames) - self.sequence_length + 1, self.stride):
                    sequence_frames = seq_frames[i:i + self.sequence_length]

                    # Check if all frames exist
                    missing_frames = [frame for frame in sequence_frames if not os.path.exists(frame['image_path'])]
                    if missing_frames:
                        skipped_sequences += 1
                    else:
                        sequences.append({
                            'frames': sequence_frames,
                            'sequence_id': f"{location}_{seq_id}_{i}",
                            'location': location
                        })

        if skipped_sequences > 0:
            logger.info(f"Created {len(sequences)} sequences, skipped {skipped_sequences} due to missing images")
        return sequences

    def __len__(self) -> int:
        """Return the number of sequences."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, List]]:
        """Get a sequence of frames and their annotations."""
        sequence = self.sequences[idx]

        images = []
        targets = []

        for frame_data in sequence['frames']:
            # Load image
            image = cv2.imread(frame_data['image_path'])
            if image is None:
                raise RuntimeError(f"Failed to load image: {frame_data['image_path']}. Image is missing or corrupted.")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Apply transforms
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
                bboxes = frame_data['bboxes']
            else:
                bboxes = frame_data['bboxes']

            # Convert to tensors
            image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            images.append(image_tensor)

            # Convert bboxes to YOLO format (normalized)
            h, w = image.shape[:2]
            yolo_targets = []
            for bbox in bboxes:
                x_center = bbox['x'] / w
                y_center = bbox['y'] / h
                width = bbox['w'] / w
                height = bbox['h'] / h
                yolo_targets.append([0, x_center, y_center, width, height])  # class 0 for fish

            targets.append(torch.tensor(yolo_targets, dtype=torch.float32))

        # Stack images and targets
        images = torch.stack(images)  # [T, C, H, W]
        targets = targets  # List of [N, 5] tensors

        return {
            'images': images,
            'targets': targets,
            'sequence_id': sequence['sequence_id'],
            'location': sequence['location']
        }


class FishDataLoader:
    """Data loader factory for fish counting dataset."""

    @staticmethod
    def get_transforms(split: str = 'train') -> A.Compose:
        """Get data augmentation transforms."""
        if split == 'train':
            transforms = A.Compose([
                A.Resize(*config.get('data.image_size', [640, 640])),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussianBlur(p=0.1),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            transforms = A.Compose([
                A.Resize(*config.get('data.image_size', [640, 640])),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        return transforms

    @staticmethod
    def create_dataloader(
        image_root: str,
        annotation_root: str,
        split: str = 'train',
        locations: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        sequence_length: Optional[int] = None,
        fish_class_id: int = 0
    ) -> DataLoader:
        """Create a data loader for the specified split."""

        batch_size = batch_size or config.get('data.batch_size', 16)
        num_workers = num_workers or config.get('data.num_workers', 4)
        sequence_length = sequence_length or config.get('data.sequence_length', 10)

        transforms = FishDataLoader.get_transforms(split)

        dataset = FishDataset(
            image_root=image_root,
            annotation_root=annotation_root,
            split=split,
            locations=locations,
            transform=transforms,
            sequence_length=sequence_length,
            fish_class_id=fish_class_id
        )

        # Validate that annotations were loaded
        if len(dataset) == 0:
            logger.warning(f"No sequences loaded for {split} split! Check annotation paths and class ID {fish_class_id}")
            logger.warning(f"Annotation root: {annotation_root}")
            logger.warning(f"Locations: {locations}")
        else:
            logger.info(f"Loaded {len(dataset)} sequences for {split} split")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            collate_fn=FishDataLoader.collate_fn,
            pin_memory=True
        )

        return dataloader

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
        """Custom collate function for variable-length targets."""
        images = []
        targets = []
        sequence_ids = []
        locations = []

        for item in batch:
            images.append(item['images'])
            targets.append(item['targets'])
            sequence_ids.append(item['sequence_id'])
            locations.append(item['location'])

        # Stack images: [B, T, C, H, W]
        images = torch.stack(images)

        return {
            'images': images,
            'targets': targets,  # List of lists of tensors
            'sequence_ids': sequence_ids,
            'locations': locations
        }