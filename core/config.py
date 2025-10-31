import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import json


class Config:
    """Configuration management for the Fish Counting project."""

    def __init__(self, config_file: Optional[str] = None):
        self._config: Dict[str, Any] = {}
        self.config_file = config_file or self._get_default_config_path()

        # Load default configuration
        self._load_defaults()

        # Load from file if exists
        if os.path.exists(self.config_file):
            self._load_from_file()

    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        project_root = Path(__file__).parent.parent
        return str(project_root / 'config.yaml')

    def _load_defaults(self):
        """Load default configuration values."""
        self._config.update({
            # Data configuration
            'data': {
                'root_dir': 'data',
                'train_split': 0.7,
                'val_split': 0.2,
                'test_split': 0.1,
                'batch_size': 16,
                'num_workers': 4,
                'image_size': [640, 640],
                'sequence_length': 10,
            },

            # Model configuration
            'model': {
                'name': 'innovative_yolo',
                'backbone': 'custom_lightweight',
                'num_classes': 1,  # fish
                'input_channels': 3,
                'temporal_attention': True,
                'sonar_optimization': True,
                'confidence_threshold': 0.25,
                'iou_threshold': 0.45,
            },

            # Training configuration
            'training': {
                'epochs': 100,
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'momentum': 0.937,
                'warmup_epochs': 3,
                'patience': 10,
                'save_freq': 10,
                'resume': False,
                'pretrained_weights': 'yolov8n.pt',
            },

            # Evaluation configuration
            'evaluation': {
                'metrics': ['precision', 'recall', 'mAP', 'count_accuracy'],
                'iou_thresholds': [0.5, 0.75],
                'confidence_range': [0.1, 0.9],
            },

            # Logging configuration
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'log_to_file': True,
                'log_dir': 'logs',
            },

            # Device configuration
            'device': {
                'gpu': True,
                'gpu_id': 0,
                'mixed_precision': True,
            }
        })

    def _load_from_file(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_file, 'r') as f:
                file_config = yaml.safe_load(f) or {}
            self._config = self._deep_update(self._config, file_config)
        except Exception as e:
            print(f"Warning: Failed to load config from {self.config_file}: {e}")

    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Deep update a dictionary."""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                base_dict[key] = self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
        return base_dict

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by dot-separated key."""
        keys = key.split('.')
        value = self._config
        try:
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            return default

    def set(self, key: str, value: Any):
        """Set a configuration value by dot-separated key."""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def save(self, filepath: Optional[str] = None):
        """Save current configuration to file."""
        filepath = filepath or self.config_file
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        """Return a copy of the configuration as a dictionary."""
        return self._config.copy()

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-like assignment."""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the configuration."""
        try:
            self.get(key)
            return True
        except KeyError:
            return False


# Global configuration instance
config = Config()