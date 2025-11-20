"""
Configuration management for Google Cloud MLOps Pipeline.

This module handles all project configuration including:
- Google Cloud Platform settings (project ID, region, credentials)
- Vertex AI configuration (training, deployment, endpoints)
- Storage configuration (GCS buckets, paths)
- Model and training parameters
- Pipeline orchestration settings
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class GCPConfig:
    """Google Cloud Platform configuration."""
    project_id: str = "your-gcp-project-id"  # Replace with your project
    region: str = "us-central1"
    zone: str = "us-central1-a"
    credentials_path: Optional[str] = None

@dataclass
class StorageConfig:
    """Cloud Storage configuration."""
    bucket_name: str = "your-mlops-bucket"  # Replace with your bucket
    data_path: str = "data/"
    models_path: str = "models/"
    pipelines_path: str = "pipelines/"
    outputs_path: str = "outputs/"
    
@dataclass
class VertexAIConfig:
    """Vertex AI configuration."""
    location: str = "us-central1"
    staging_bucket: str = "your-mlops-bucket"  # Replace with your bucket
    service_account: Optional[str] = None
    network: Optional[str] = None
    
@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "iris-classifier"
    framework: str = "scikit-learn"
    version: str = "v1.0"
    
@dataclass
class TrainingConfig:
    """Training configuration."""
    dataset: str = "iris"
    test_size: float = 0.2
    random_state: int = 42
    max_trials: int = 10
    
@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    name: str = "iris-mlops-pipeline"
    description: str = "End-to-end Iris classification pipeline"
    
@dataclass
class Config:
    """Main configuration class."""
    project_name: str = "mlops-demo"
    project_description: str = "End-to-end MLOps pipeline with Google Vertex AI"
    project_version: str = "1.0.0"
    
    gcp: GCPConfig = field(default_factory=GCPConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    vertex_ai: VertexAIConfig = field(default_factory=VertexAIConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file. If None, uses default.
        
    Returns:
        Config: Configuration object with all settings.
    """
    if config_path is None:
        # Default to configs/config.yaml relative to project root
        project_root = Path(__file__).parent.parent
        config_path_obj = project_root / "configs" / "config.yaml"
    else:
        config_path_obj = Path(config_path)
    
    # Create default config
    config = Config()
    
    if config_path_obj.exists():
        try:
            with open(config_path_obj, 'r') as f:
                yaml_config = yaml.safe_load(f)
                
            # Update config with values from YAML
            if 'gcp' in yaml_config:
                gcp_data = yaml_config['gcp']
                config.gcp.project_id = gcp_data.get('project_id', config.gcp.project_id)
                config.gcp.region = gcp_data.get('region', config.gcp.region)
                config.gcp.zone = gcp_data.get('zone', config.gcp.zone)
                
            if 'storage' in yaml_config:
                storage_data = yaml_config['storage']
                config.storage.bucket_name = storage_data.get('bucket_name', config.storage.bucket_name)
                config.storage.data_path = storage_data.get('data_path', config.storage.data_path)
                config.storage.models_path = storage_data.get('models_path', config.storage.models_path)
                
            if 'vertex_ai' in yaml_config:
                vertex_data = yaml_config['vertex_ai']
                config.vertex_ai.location = vertex_data.get('location', config.vertex_ai.location)
                config.vertex_ai.staging_bucket = vertex_data.get('staging_bucket', config.vertex_ai.staging_bucket)
                
            if 'model' in yaml_config:
                model_data = yaml_config['model']
                config.model.name = model_data.get('name', config.model.name)
                config.model.framework = model_data.get('framework', config.model.framework)
                
            if 'training' in yaml_config:
                training_data = yaml_config['training']
                config.training.dataset = training_data.get('dataset', config.training.dataset)
                config.training.test_size = training_data.get('test_size', config.training.test_size)
                
            logger.info(f"Configuration loaded from {config_path_obj}")
            
        except Exception as e:
            logger.warning(f"Could not load config from {config_path_obj}: {e}")
            logger.info("Using default configuration")
    else:
        logger.info("Configuration file not found, using defaults")
        
    # Override with environment variables if set
    config.gcp.project_id = os.getenv('GCP_PROJECT_ID', config.gcp.project_id)
    config.gcp.region = os.getenv('GCP_REGION', config.gcp.region)
    config.storage.bucket_name = os.getenv('GCS_BUCKET', config.storage.bucket_name)
    
    return config

def get_config() -> Config:
    """Get the global configuration instance."""
    return load_config()

# Global config instance
config = get_config()
