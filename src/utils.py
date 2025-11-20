"""
Utility functions for Google Cloud MLOps Pipeline.

This module provides common utilities including:
- Logging setup and configuration
- File I/O operations
- Data validation helpers
- Performance monitoring
- Error handling utilities
"""

import os
import sys
import logging
import json
import pickle
import joblib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import numpy as np
import pandas as pd

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration for the MLOps pipeline.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional log file path. If None, logs to console only.
        format_string: Custom format string for log messages.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Add file handler if log_file is specified
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        logging.getLogger().addHandler(file_handler)
    
    logger = logging.getLogger(__name__)
    logger.info("Logging configured successfully")
    return logger

def save_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save.
        file_path: Path to save the JSON file.
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.
    
    Args:
        file_path: Path to the JSON file.
        
    Returns:
        Dict[str, Any]: Loaded dictionary.
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def save_model(model: Any, file_path: str) -> None:
    """
    Save model using joblib.
    
    Args:
        model: Model object to save.
        file_path: Path to save the model file.
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, file_path)

def load_model(file_path: str) -> Any:
    """
    Load model using joblib.
    
    Args:
        file_path: Path to the model file.
        
    Returns:
        Any: Loaded model object.
    """
    return joblib.load(file_path)

def save_pickle(data: Any, file_path: str) -> None:
    """
    Save data using pickle.
    
    Args:
        data: Data to save.
        file_path: Path to save the pickle file.
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file_path: str) -> Any:
    """
    Load data using pickle.
    
    Args:
        file_path: Path to the pickle file.
        
    Returns:
        Any: Loaded data.
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def create_directories(*dirs: str) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        dirs: Directory paths to create.
    """
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def get_timestamp(format_string: str = "%Y%m%d_%H%M%S") -> str:
    """
    Get current timestamp as string.
    
    Args:
        format_string: Format for timestamp string.
        
    Returns:
        str: Formatted timestamp.
    """
    return datetime.now().strftime(format_string)

def validate_file_exists(file_path: str) -> bool:
    """
    Check if file exists.
    
    Args:
        file_path: Path to check.
        
    Returns:
        bool: True if file exists, False otherwise.
    """
    return Path(file_path).exists()

def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to the file.
        
    Returns:
        int: File size in bytes.
    """
    return Path(file_path).stat().st_size

def format_bytes(size: int) -> str:
    """
    Format bytes to human readable string.
    
    Args:
        size: Size in bytes.
        
    Returns:
        str: Formatted size string.
    """
    size_float = float(size)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_float < 1024.0:
            return f"{size_float:.1f} {unit}"
        size_float /= 1024.0
    return f"{size_float:.1f} PB"

def calculate_memory_usage() -> Dict[str, str]:
    """
    Calculate current memory usage.
    
    Returns:
        Dict[str, str]: Memory usage statistics.
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss': format_bytes(memory_info.rss),
            'vms': format_bytes(memory_info.vms),
            'percent': f"{process.memory_percent():.1f}%"
        }
    except ImportError:
        return {'message': 'psutil not available'}

def timer(func):
    """
    Decorator to time function execution.
    
    Args:
        func: Function to time.
        
    Returns:
        Function wrapper with timing.
    """
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger = logging.getLogger(__name__)
        logger.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

class ProgressTracker:
    """Simple progress tracking utility."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.logger = logging.getLogger(__name__)
        
    def update(self, increment: int = 1) -> None:
        """Update progress."""
        self.current += increment
        percentage = (self.current / self.total) * 100
        self.logger.info(f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%)")
        
    def reset(self) -> None:
        """Reset progress."""
        self.current = 0

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers.
    
    Args:
        a: Numerator.
        b: Denominator.
        default: Default value if division by zero.
        
    Returns:
        float: Division result or default value.
    """
    return a / b if b != 0 else default

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary.
    
    Args:
        d: Dictionary to flatten.
        parent_key: Parent key prefix.
        sep: Separator for nested keys.
        
    Returns:
        Dict[str, Any]: Flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# Initialize logging
logger = setup_logging()
