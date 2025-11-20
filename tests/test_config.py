"""
Tests for configuration management.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from src.config import Config


def test_config_loading():
    """Test configuration file loading."""
    # Create temporary config file
    config_data = {
        'gcp': {
            'project_id': 'test-project',
            'region': 'us-central1'
        },
        'model': {
            'name': 'test-model'
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_config_path = f.name
    
    try:
        config = Config(config_path=temp_config_path)
        
        assert config.get('gcp.project_id') == 'test-project'
        assert config.get('gcp.region') == 'us-central1'
        assert config.get('model.name') == 'test-model'
        assert config.get('nonexistent.key', 'default') == 'default'
        
    finally:
        Path(temp_config_path).unlink()


def test_config_properties():
    """Test configuration property shortcuts."""
    config_data = {
        'gcp': {
            'project_id': 'test-project',
            'region': 'us-west1'
        },
        'storage': {
            'bucket_name': 'test-bucket'
        },
        'model': {
            'name': 'test-model'
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_config_path = f.name
    
    try:
        config = Config(config_path=temp_config_path)
        
        assert config.project_id == 'test-project'
        assert config.region == 'us-west1'
        assert config.bucket_name == 'test-bucket'
        assert config.model_name == 'test-model'
        
    finally:
        Path(temp_config_path).unlink()


def test_config_file_not_found():
    """Test behavior when config file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        Config(config_path='nonexistent.yaml')
