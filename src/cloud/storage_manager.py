"""
Cloud Storage Integration Module

This module provides utilities for managing model artifacts, training data,
and metadata in Google Cloud Storage, integrating with the MLOps pipeline.

Classes:
    CloudStorageManager: Main class for GCS operations
    ArtifactUploader: Specialized uploader for model artifacts
    ModelMetadataManager: Manager for model metadata in cloud storage

Author: MLOps Team
Version: 1.0.0
"""

import os
import json
import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import subprocess
import hashlib

# Local imports
from ..config import Config
from ..utils import setup_logging

logger = logging.getLogger(__name__)

# Check for Google Cloud SDK availability
try:
    result = subprocess.run(['gcloud', 'version'], 
                          capture_output=True, text=True, timeout=10)
    GCLOUD_AVAILABLE = result.returncode == 0
except (subprocess.TimeoutExpired, FileNotFoundError):
    GCLOUD_AVAILABLE = False

# Check for Google Cloud Storage SDK
storage_module = None
try:
    from google.cloud import storage as storage_module
    GCS_SDK_AVAILABLE = True
except ImportError:
    GCS_SDK_AVAILABLE = False
    logging.warning("Google Cloud Storage SDK not available. "
                   "Install with: pip install google-cloud-storage")


@dataclass
class ArtifactMetadata:
    """Metadata for ML artifacts stored in cloud storage."""
    name: str
    type: str  # 'model', 'dataset', 'evaluation', 'config'
    version: str
    created_at: str
    size_bytes: int
    checksum: str
    tags: Dict[str, str]
    gcs_path: str
    local_path: Optional[str] = None


class CloudStorageManager:
    """
    Manager for cloud storage operations with Google Cloud Storage.
    
    Handles upload, download, and management of ML artifacts including models,
    datasets, and training results.
    """
    
    def __init__(self, project_id: str, bucket_name: str, 
                 credentials_path: Optional[str] = None):
        """
        Initialize cloud storage manager.
        
        Args:
            project_id: Google Cloud project ID
            bucket_name: GCS bucket name for artifact storage
            credentials_path: Path to service account credentials (optional)
        """
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.credentials_path = credentials_path
        
        # Initialize storage client if SDK is available
        self.client = None
        self.bucket = None
        
        if GCS_SDK_AVAILABLE and storage_module:
            try:
                if credentials_path:
                    self.client = storage_module.Client.from_service_account_json(
                        credentials_path, project=project_id
                    )
                else:
                    self.client = storage_module.Client(project=project_id)
                
                self.bucket = self.client.bucket(bucket_name)
                logger.info(f"GCS client initialized for bucket: {bucket_name}")
                
            except Exception as e:
                logger.warning(f"Failed to initialize GCS client: {e}")
                self.client = None
                self.bucket = None
        
        # Check if gcloud CLI is available as fallback
        if not self.client and not GCLOUD_AVAILABLE:
            logger.warning("Neither GCS SDK nor gcloud CLI available")
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _run_gcloud_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run a gcloud storage command."""
        full_cmd = ['gcloud', 'storage'] + cmd
        logger.debug(f"Running: {' '.join(full_cmd)}")
        
        result = subprocess.run(
            full_cmd, 
            capture_output=True, 
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            logger.error(f"Command failed: {' '.join(full_cmd)}")
            logger.error(f"Error: {result.stderr}")
            raise subprocess.CalledProcessError(
                result.returncode, full_cmd, result.stdout, result.stderr
            )
        
        return result
    
    def upload_file(self, local_path: str, gcs_path: str, 
                   metadata: Optional[Dict[str, str]] = None) -> str:
        """
        Upload a file to Google Cloud Storage.
        
        Args:
            local_path: Local file path
            gcs_path: GCS destination path (without gs:// prefix)
            metadata: Optional metadata dictionary
            
        Returns:
            Full GCS URI
        """
        full_gcs_path = f"gs://{self.bucket_name}/{gcs_path}"
        
        if self.client and self.bucket:
            # Use SDK
            try:
                blob = self.bucket.blob(gcs_path)
                
                # Set metadata
                if metadata:
                    blob.metadata = metadata
                
                # Upload file
                blob.upload_from_filename(local_path)
                
                logger.info(f"Uploaded {local_path} to {full_gcs_path}")
                return full_gcs_path
                
            except Exception as e:
                logger.error(f"SDK upload failed: {e}")
                # Fall back to gcloud CLI
        
        # Use gcloud CLI
        if GCLOUD_AVAILABLE:
            try:
                cmd = ['cp', local_path, full_gcs_path]
                self._run_gcloud_command(cmd)
                
                logger.info(f"Uploaded {local_path} to {full_gcs_path}")
                return full_gcs_path
                
            except Exception as e:
                logger.error(f"CLI upload failed: {e}")
                raise
        
        raise RuntimeError("No GCS upload method available")
    
    def download_file(self, gcs_path: str, local_path: str) -> str:
        """
        Download a file from Google Cloud Storage.
        
        Args:
            gcs_path: GCS path (with or without gs:// prefix)
            local_path: Local destination path
            
        Returns:
            Local file path
        """
        # Normalize GCS path
        if not gcs_path.startswith('gs://'):
            gcs_path = f"gs://{self.bucket_name}/{gcs_path}"
        
        # Ensure local directory exists
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        
        if self.client and self.bucket:
            # Use SDK
            try:
                # Extract blob name from full path
                blob_name = gcs_path.replace(f"gs://{self.bucket_name}/", "")
                blob = self.bucket.blob(blob_name)
                blob.download_to_filename(local_path)
                
                logger.info(f"Downloaded {gcs_path} to {local_path}")
                return local_path
                
            except Exception as e:
                logger.error(f"SDK download failed: {e}")
                # Fall back to gcloud CLI
        
        # Use gcloud CLI
        if GCLOUD_AVAILABLE:
            try:
                cmd = ['cp', gcs_path, local_path]
                self._run_gcloud_command(cmd)
                
                logger.info(f"Downloaded {gcs_path} to {local_path}")
                return local_path
                
            except Exception as e:
                logger.error(f"CLI download failed: {e}")
                raise
        
        raise RuntimeError("No GCS download method available")
    
    def list_objects(self, prefix: str = "") -> List[str]:
        """
        List objects in the bucket with optional prefix.
        
        Args:
            prefix: Object prefix filter
            
        Returns:
            List of object names
        """
        if self.client and self.bucket:
            # Use SDK
            try:
                blobs = self.bucket.list_blobs(prefix=prefix)
                return [blob.name for blob in blobs]
                
            except Exception as e:
                logger.error(f"SDK list failed: {e}")
                # Fall back to gcloud CLI
        
        # Use gcloud CLI
        if GCLOUD_AVAILABLE:
            try:
                bucket_path = f"gs://{self.bucket_name}/"
                if prefix:
                    bucket_path += prefix
                
                cmd = ['ls', bucket_path]
                result = self._run_gcloud_command(cmd)
                
                # Parse output to get object names
                lines = result.stdout.strip().split('\n')
                objects = []
                for line in lines:
                    if line.strip() and not line.endswith('/'):
                        # Extract object name from full path
                        obj_name = line.replace(f"gs://{self.bucket_name}/", "")
                        objects.append(obj_name)
                
                return objects
                
            except Exception as e:
                logger.error(f"CLI list failed: {e}")
                return []
        
        return []
    
    def delete_object(self, gcs_path: str) -> bool:
        """
        Delete an object from Google Cloud Storage.
        
        Args:
            gcs_path: GCS path (with or without gs:// prefix)
            
        Returns:
            True if successful, False otherwise
        """
        # Normalize GCS path
        if not gcs_path.startswith('gs://'):
            gcs_path = f"gs://{self.bucket_name}/{gcs_path}"
        
        if self.client and self.bucket:
            # Use SDK
            try:
                blob_name = gcs_path.replace(f"gs://{self.bucket_name}/", "")
                blob = self.bucket.blob(blob_name)
                blob.delete()
                
                logger.info(f"Deleted {gcs_path}")
                return True
                
            except Exception as e:
                logger.error(f"SDK delete failed: {e}")
                # Fall back to gcloud CLI
        
        # Use gcloud CLI
        if GCLOUD_AVAILABLE:
            try:
                cmd = ['rm', gcs_path]
                self._run_gcloud_command(cmd)
                
                logger.info(f"Deleted {gcs_path}")
                return True
                
            except Exception as e:
                logger.error(f"CLI delete failed: {e}")
                return False
        
        return False


class ArtifactUploader:
    """
    Specialized uploader for ML artifacts with metadata tracking.
    """
    
    def __init__(self, storage_manager: CloudStorageManager, 
                 artifacts_prefix: str = "artifacts"):
        """
        Initialize artifact uploader.
        
        Args:
            storage_manager: CloudStorageManager instance
            artifacts_prefix: Prefix for all artifacts in bucket
        """
        self.storage_manager = storage_manager
        self.artifacts_prefix = artifacts_prefix
        self.metadata_cache = {}
    
    def upload_model(self, model_path: str, model_name: str, 
                    version: str, tags: Optional[Dict[str, str]] = None) -> ArtifactMetadata:
        """
        Upload a trained model with metadata.
        
        Args:
            model_path: Local path to model file
            model_name: Name of the model
            version: Model version
            tags: Optional metadata tags
            
        Returns:
            ArtifactMetadata object
        """
        # Calculate metadata
        file_size = os.path.getsize(model_path)
        checksum = self.storage_manager._calculate_checksum(model_path)
        timestamp = datetime.now().isoformat()
        
        # Define GCS path
        gcs_path = f"{self.artifacts_prefix}/models/{model_name}/{version}/model.pkl"
        
        # Upload model file
        full_gcs_path = self.storage_manager.upload_file(
            model_path, 
            gcs_path,
            metadata={
                'model_name': model_name,
                'version': version,
                'created_at': timestamp,
                'checksum': checksum
            }
        )
        
        # Create metadata object
        metadata = ArtifactMetadata(
            name=model_name,
            type='model',
            version=version,
            created_at=timestamp,
            size_bytes=file_size,
            checksum=checksum,
            tags=tags or {},
            gcs_path=full_gcs_path,
            local_path=model_path
        )
        
        # Upload metadata file
        metadata_path = f"{self.artifacts_prefix}/models/{model_name}/{version}/metadata.json"
        self._upload_metadata(metadata, metadata_path)
        
        # Cache metadata
        self.metadata_cache[f"{model_name}:{version}"] = metadata
        
        logger.info(f"Model uploaded: {model_name} v{version} to {full_gcs_path}")
        return metadata
    
    def upload_dataset(self, dataset_path: str, dataset_name: str,
                      tags: Optional[Dict[str, str]] = None) -> ArtifactMetadata:
        """
        Upload a dataset with metadata.
        
        Args:
            dataset_path: Local path to dataset file
            dataset_name: Name of the dataset
            tags: Optional metadata tags
            
        Returns:
            ArtifactMetadata object
        """
        file_size = os.path.getsize(dataset_path)
        checksum = self.storage_manager._calculate_checksum(dataset_path)
        timestamp = datetime.now().isoformat()
        version = timestamp.split('T')[0].replace('-', '')  # YYYYMMDD
        
        # Define GCS path
        file_ext = Path(dataset_path).suffix
        gcs_path = f"{self.artifacts_prefix}/datasets/{dataset_name}/{version}/data{file_ext}"
        
        # Upload dataset
        full_gcs_path = self.storage_manager.upload_file(
            dataset_path,
            gcs_path,
            metadata={
                'dataset_name': dataset_name,
                'version': version,
                'created_at': timestamp,
                'checksum': checksum
            }
        )
        
        # Create metadata
        metadata = ArtifactMetadata(
            name=dataset_name,
            type='dataset',
            version=version,
            created_at=timestamp,
            size_bytes=file_size,
            checksum=checksum,
            tags=tags or {},
            gcs_path=full_gcs_path,
            local_path=dataset_path
        )
        
        # Upload metadata
        metadata_path = f"{self.artifacts_prefix}/datasets/{dataset_name}/{version}/metadata.json"
        self._upload_metadata(metadata, metadata_path)
        
        logger.info(f"Dataset uploaded: {dataset_name} to {full_gcs_path}")
        return metadata
    
    def upload_training_results(self, results_path: str, experiment_name: str,
                               tags: Optional[Dict[str, str]] = None) -> ArtifactMetadata:
        """
        Upload training results and evaluation metrics.
        
        Args:
            results_path: Local path to results file
            experiment_name: Name of the experiment
            tags: Optional metadata tags
            
        Returns:
            ArtifactMetadata object
        """
        file_size = os.path.getsize(results_path)
        checksum = self.storage_manager._calculate_checksum(results_path)
        timestamp = datetime.now().isoformat()
        
        # Define GCS path
        file_ext = Path(results_path).suffix
        gcs_path = f"{self.artifacts_prefix}/experiments/{experiment_name}/{timestamp}/results{file_ext}"
        
        # Upload results
        full_gcs_path = self.storage_manager.upload_file(
            results_path,
            gcs_path,
            metadata={
                'experiment_name': experiment_name,
                'created_at': timestamp,
                'checksum': checksum,
                'type': 'training_results'
            }
        )
        
        # Create metadata
        metadata = ArtifactMetadata(
            name=experiment_name,
            type='evaluation',
            version=timestamp,
            created_at=timestamp,
            size_bytes=file_size,
            checksum=checksum,
            tags=tags or {},
            gcs_path=full_gcs_path,
            local_path=results_path
        )
        
        # Upload metadata
        metadata_path = f"{self.artifacts_prefix}/experiments/{experiment_name}/{timestamp}/metadata.json"
        self._upload_metadata(metadata, metadata_path)
        
        logger.info(f"Training results uploaded: {experiment_name} to {full_gcs_path}")
        return metadata
    
    def _upload_metadata(self, metadata: ArtifactMetadata, gcs_path: str):
        """Upload metadata as JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'name': metadata.name,
                'type': metadata.type,
                'version': metadata.version,
                'created_at': metadata.created_at,
                'size_bytes': metadata.size_bytes,
                'checksum': metadata.checksum,
                'tags': metadata.tags,
                'gcs_path': metadata.gcs_path,
                'local_path': metadata.local_path
            }, f, indent=2)
            
            temp_path = f.name
        
        try:
            self.storage_manager.upload_file(temp_path, gcs_path)
        finally:
            os.unlink(temp_path)
    
    def list_models(self, model_name: Optional[str] = None) -> List[ArtifactMetadata]:
        """
        List available models in cloud storage.
        
        Args:
            model_name: Optional model name filter
            
        Returns:
            List of model metadata
        """
        prefix = f"{self.artifacts_prefix}/models/"
        if model_name:
            prefix += f"{model_name}/"
        
        objects = self.storage_manager.list_objects(prefix)
        
        # Filter for metadata files
        metadata_files = [obj for obj in objects if obj.endswith('/metadata.json')]
        
        models = []
        for metadata_file in metadata_files:
            try:
                # Download and parse metadata
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    temp_path = f.name
                
                self.storage_manager.download_file(metadata_file, temp_path)
                
                with open(temp_path, 'r') as f:
                    metadata_dict = json.load(f)
                
                metadata = ArtifactMetadata(**metadata_dict)
                models.append(metadata)
                
                os.unlink(temp_path)
                
            except Exception as e:
                logger.warning(f"Failed to load metadata from {metadata_file}: {e}")
                continue
        
        return sorted(models, key=lambda x: x.created_at, reverse=True)
    
    def download_model(self, model_name: str, version: str, 
                      local_path: str) -> ArtifactMetadata:
        """
        Download a model from cloud storage.
        
        Args:
            model_name: Name of the model
            version: Model version
            local_path: Local destination path
            
        Returns:
            ArtifactMetadata object
        """
        # Download metadata first
        metadata_path = f"{self.artifacts_prefix}/models/{model_name}/{version}/metadata.json"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_metadata_path = f.name
        
        try:
            self.storage_manager.download_file(metadata_path, temp_metadata_path)
            
            with open(temp_metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            
            metadata = ArtifactMetadata(**metadata_dict)
            
            # Download model file
            model_gcs_path = f"{self.artifacts_prefix}/models/{model_name}/{version}/model.pkl"
            self.storage_manager.download_file(model_gcs_path, local_path)
            
            # Update local path in metadata
            metadata.local_path = local_path
            
            logger.info(f"Model downloaded: {model_name} v{version} to {local_path}")
            return metadata
            
        finally:
            os.unlink(temp_metadata_path)


class ModelMetadataManager:
    """
    Manager for model metadata and versioning in cloud storage.
    """
    
    def __init__(self, storage_manager: CloudStorageManager):
        """
        Initialize metadata manager.
        
        Args:
            storage_manager: CloudStorageManager instance
        """
        self.storage_manager = storage_manager
        self.registry_path = "model_registry/registry.json"
        self._registry_cache = None
    
    def register_model(self, metadata: ArtifactMetadata, 
                      description: str = "", 
                      performance_metrics: Optional[Dict[str, float]] = None) -> bool:
        """
        Register a model in the central registry.
        
        Args:
            metadata: ArtifactMetadata object
            description: Model description
            performance_metrics: Performance metrics dictionary
            
        Returns:
            True if successful
        """
        try:
            # Load current registry
            registry = self._load_registry()
            
            # Create registry entry
            entry = {
                'name': metadata.name,
                'version': metadata.version,
                'type': metadata.type,
                'created_at': metadata.created_at,
                'size_bytes': metadata.size_bytes,
                'checksum': metadata.checksum,
                'gcs_path': metadata.gcs_path,
                'tags': metadata.tags,
                'description': description,
                'performance_metrics': performance_metrics or {},
                'registered_at': datetime.now().isoformat()
            }
            
            # Add to registry
            if metadata.name not in registry:
                registry[metadata.name] = {}
            
            registry[metadata.name][metadata.version] = entry
            
            # Save updated registry
            self._save_registry(registry)
            
            logger.info(f"Model registered: {metadata.name} v{metadata.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return False
    
    def get_latest_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest version of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Latest model entry or None
        """
        registry = self._load_registry()
        
        if model_name not in registry:
            return None
        
        versions = registry[model_name]
        if not versions:
            return None
        
        # Sort by created_at and return latest
        latest_version = max(versions.items(), key=lambda x: x[1]['created_at'])
        return latest_version[1]
    
    def list_registered_models(self) -> Dict[str, List[str]]:
        """
        List all registered models and their versions.
        
        Returns:
            Dictionary mapping model names to version lists
        """
        registry = self._load_registry()
        return {name: list(versions.keys()) for name, versions in registry.items()}
    
    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load the model registry from cloud storage."""
        if self._registry_cache is not None:
            return self._registry_cache
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                temp_path = f.name
            
            self.storage_manager.download_file(self.registry_path, temp_path)
            
            with open(temp_path, 'r') as f:
                registry = json.load(f)
            
            os.unlink(temp_path)
            self._registry_cache = registry
            return registry
            
        except Exception:
            # Registry doesn't exist, create empty one
            logger.info("Creating new model registry")
            return {}
    
    def _save_registry(self, registry: Dict[str, Dict[str, Any]]):
        """Save the model registry to cloud storage."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(registry, f, indent=2)
            temp_path = f.name
        
        try:
            self.storage_manager.upload_file(temp_path, self.registry_path)
            self._registry_cache = registry
        finally:
            os.unlink(temp_path)


def create_cloud_storage_manager(project_id: str, bucket_name: str) -> CloudStorageManager:
    """
    Create a configured CloudStorageManager instance.
    
    Args:
        project_id: Google Cloud project ID
        bucket_name: GCS bucket name
        
    Returns:
        CloudStorageManager instance
    """
    return CloudStorageManager(project_id, bucket_name)


def create_artifact_uploader(storage_manager: CloudStorageManager) -> ArtifactUploader:
    """
    Create an ArtifactUploader instance.
    
    Args:
        storage_manager: CloudStorageManager instance
        
    Returns:
        ArtifactUploader instance
    """
    return ArtifactUploader(storage_manager)


def create_metadata_manager(storage_manager: CloudStorageManager) -> ModelMetadataManager:
    """
    Create a ModelMetadataManager instance.
    
    Args:
        storage_manager: CloudStorageManager instance
        
    Returns:
        ModelMetadataManager instance
    """
    return ModelMetadataManager(storage_manager)
