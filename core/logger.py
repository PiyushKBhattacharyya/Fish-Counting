import logging
import logging.handlers
from pathlib import Path
from typing import Optional
import sys
import os

from .config import config


class Logger:
    """Centralized logging configuration for the Fish Counting project."""

    _instance: Optional['Logger'] = None
    _logger: Optional[logging.Logger] = None

    def __new__(cls) -> 'Logger':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._logger is None:
            self._setup_logger()

    def _setup_logger(self):
        """Set up the logger with appropriate handlers and formatters."""
        self._logger = logging.getLogger('fish_counting')
        self._logger.setLevel(getattr(logging, config.get('logging.level', 'INFO')))

        # Remove existing handlers
        self._logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(
            config.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

        # File handler (if enabled)
        if config.get('logging.log_to_file', True):
            log_dir = config.get('logging.log_dir', 'logs')
            os.makedirs(log_dir, exist_ok=True)

            log_file = Path(log_dir) / 'fish_counting.log'
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5
            )
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """Get a logger instance with optional name."""
        if name:
            return self._logger.getChild(name)
        return self._logger

    def debug(self, message: str, *args, **kwargs):
        """Log a debug message."""
        self._logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        """Log an info message."""
        self._logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        """Log a warning message."""
        self._logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        """Log an error message."""
        self._logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs):
        """Log a critical message."""
        self._logger.critical(message, *args, **kwargs)

    def exception(self, message: str, *args, **kwargs):
        """Log an exception with traceback."""
        self._logger.exception(message, *args, **kwargs)

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """Log training/validation metrics."""
        step_str = f"Step {step}: " if step is not None else ""
        metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        self._logger.info(f"{step_str}Metrics - {metrics_str}")

    def log_epoch_summary(self, epoch: int, train_metrics: dict, val_metrics: Optional[dict] = None):
        """Log epoch summary with training and validation metrics."""
        self._logger.info(f"=== Epoch {epoch} Summary ===")
        self.log_metrics(train_metrics, step=None)

        if val_metrics:
            self._logger.info("Validation Metrics:")
            self.log_metrics(val_metrics, step=None)

    def set_level(self, level: str):
        """Set the logging level."""
        self._logger.setLevel(getattr(logging, level.upper()))

    def add_file_handler(self, filepath: str, level: Optional[str] = None):
        """Add an additional file handler."""
        formatter = logging.Formatter(
            config.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )

        handler = logging.FileHandler(filepath)
        handler.setFormatter(formatter)
        if level:
            handler.setLevel(getattr(logging, level.upper()))
        self._logger.addHandler(handler)


# Global logger instance
logger = Logger()