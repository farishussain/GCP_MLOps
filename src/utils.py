"""
Utility functions for the MLOps project.

This module contains common utility functions used across the project.
"""

import os
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def ensure_directory(path: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path (str): Directory path to create.
        
    Returns:
        Path: Path object for the directory.
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def list_files(directory: str, pattern: str = "*") -> List[Path]:
    """
    List files in directory matching pattern.
    
    Args:
        directory (str): Directory to search.
        pattern (str): File pattern to match.
        
    Returns:
        List[Path]: List of matching file paths.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    
    return list(dir_path.glob(pattern))


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path: Project root directory path.
    """
    return Path(__file__).parent.parent


def validate_gcp_environment() -> Dict[str, bool]:
    """
    Validate Google Cloud environment setup.
    
    Returns:
        Dict[str, bool]: Validation results for different components.
    """
    validation_results = {
        'gcloud_cli': False,
        'application_credentials': False,
        'project_id': False
    }
    
    # Check gcloud CLI
    try:
        result = subprocess.run(['gcloud', '--version'], 
                              capture_output=True, text=True)
        validation_results['gcloud_cli'] = result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    
    # Check application credentials
    if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
        cred_file = Path(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
        validation_results['application_credentials'] = cred_file.exists()
    
    # Check project ID
    validation_results['project_id'] = bool(os.environ.get('GOOGLE_CLOUD_PROJECT'))
    
    return validation_results
