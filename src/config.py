"""
Configuration management utilities for the MLOps project.

This module provides utilities for loading and managing configuration
settings across the project.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for the MLOps project."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path (str): Path to configuration file.
        """
        if config_path is None:
            # Default to config.yaml in configs directory
            project_root = Path(__file__).parent.parent
            self.config_path = project_root / "configs" / "config.yaml"
        else:
            self.config_path = Path(config_path)
        
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dict[str, Any]: Configuration dictionary.
        """
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path (str): Dot-separated key path (e.g., 'gcp.project_id').
            default (Any): Default value if key not found.
            
        Returns:
            Any: Configuration value.
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    @property
    def project_id(self) -> str:
        """Get GCP project ID."""
        return self.get('gcp.project_id')
    
    @property
    def region(self) -> str:
        """Get GCP region."""
        return self.get('gcp.region')
    
    @property
    def bucket_name(self) -> str:
        """Get GCS bucket name."""
        return self.get('storage.bucket_name')
    
    @property
    def model_name(self) -> str:
        """Get model name."""
        return self.get('model.name')


# Global configuration instance
config = Config()
