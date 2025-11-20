"""
Cloud Storage management utilities for MLOps pipeline.

This module provides Google Cloud Storage integration including:
- File upload and download
- Bucket management
- Artifact storage and retrieval
- Data versioning
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import tempfile
import shutil

# Handle optional Google Cloud dependencies
try:
    from google.cloud import storage  # type: ignore
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False
    storage = None  # type: ignore
    logging.warning("Google Cloud Storage library not available. Install with: pip install google-cloud-storage")

from ..config import get_config
from ..utils import get_timestamp

logger = logging.getLogger(__name__)

class CloudStorageManager:
    """Google Cloud Storage management utility class."""
    
    def __init__(self, bucket_name: Optional[str] = None):
        if not GOOGLE_CLOUD_AVAILABLE or storage is None:
            raise ImportError("Google Cloud Storage not available. Install with: pip install google-cloud-storage")
        
        config = get_config()
        self.bucket_name = bucket_name or config.storage.bucket_name
        self.client = storage.Client()  # type: ignore
        self.bucket = self.client.bucket(self.bucket_name)
        
        logger.info(f"Cloud Storage manager initialized for bucket: {self.bucket_name}")
    
    def upload_file(
        self,
        source_path: str,
        destination_path: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Upload a file to Cloud Storage.
        
        Args:
            source_path: Local file path to upload.
            destination_path: Destination path in the bucket.
            metadata: Optional metadata to attach to the file.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            blob = self.bucket.blob(destination_path)
            
            # Set metadata if provided
            if metadata:
                blob.metadata = metadata
            
            blob.upload_from_filename(source_path)
            
            logger.info(f"File uploaded: {source_path} -> gs://{self.bucket_name}/{destination_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload file {source_path}: {e}")
            return False
    
    def download_file(
        self,
        source_path: str,
        destination_path: str
    ) -> bool:
        """
        Download a file from Cloud Storage.
        
        Args:
            source_path: Source path in the bucket.
            destination_path: Local destination path.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            blob = self.bucket.blob(source_path)
            
            # Create destination directory if it doesn't exist
            Path(destination_path).parent.mkdir(parents=True, exist_ok=True)
            
            blob.download_to_filename(destination_path)
            
            logger.info(f"File downloaded: gs://{self.bucket_name}/{source_path} -> {destination_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download file {source_path}: {e}")
            return False
    
    def upload_directory(
        self,
        source_directory: str,
        destination_prefix: str,
        exclude_patterns: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """
        Upload a directory to Cloud Storage.
        
        Args:
            source_directory: Local directory to upload.
            destination_prefix: Destination prefix in the bucket.
            exclude_patterns: List of patterns to exclude.
            
        Returns:
            Dictionary mapping file paths to upload success status.
        """
        results = {}
        source_path = Path(source_directory)
        
        if not source_path.exists():
            logger.error(f"Source directory does not exist: {source_directory}")
            return results
        
        exclude_patterns = exclude_patterns or []
        
        for file_path in source_path.rglob('*'):
            if file_path.is_file():
                # Check if file should be excluded
                should_exclude = any(
                    pattern in str(file_path) for pattern in exclude_patterns
                )
                
                if should_exclude:
                    continue
                
                # Calculate relative path and destination
                relative_path = file_path.relative_to(source_path)
                destination_path = f"{destination_prefix}/{relative_path}".replace('\\', '/')
                
                # Upload file
                success = self.upload_file(str(file_path), destination_path)
                results[str(relative_path)] = success
        
        logger.info(f"Directory upload complete: {len(results)} files processed")
        return results
    
    def download_directory(
        self,
        source_prefix: str,
        destination_directory: str
    ) -> Dict[str, bool]:
        """
        Download a directory from Cloud Storage.
        
        Args:
            source_prefix: Source prefix in the bucket.
            destination_directory: Local destination directory.
            
        Returns:
            Dictionary mapping file paths to download success status.
        """
        results = {}
        destination_path = Path(destination_directory)
        destination_path.mkdir(parents=True, exist_ok=True)
        
        # List all blobs with the given prefix
        blobs = self.bucket.list_blobs(prefix=source_prefix)
        
        for blob in blobs:
            if not blob.name.endswith('/'):  # Skip directory markers
                # Calculate local file path
                relative_path = blob.name[len(source_prefix):].lstrip('/')
                local_file_path = destination_path / relative_path
                
                # Download file
                success = self.download_file(blob.name, str(local_file_path))
                results[relative_path] = success
        
        logger.info(f"Directory download complete: {len(results)} files processed")
        return results
    
    def list_files(
        self,
        prefix: str = "",
        max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        List files in the bucket.
        
        Args:
            prefix: Prefix to filter files.
            max_results: Maximum number of results to return.
            
        Returns:
            List of file information dictionaries.
        """
        try:
            blobs = self.bucket.list_blobs(
                prefix=prefix,
                max_results=max_results
            )
            
            files = []
            for blob in blobs:
                file_info = {
                    'name': blob.name,
                    'size': blob.size,
                    'created': blob.time_created.isoformat() if blob.time_created else None,
                    'updated': blob.updated.isoformat() if blob.updated else None,
                    'content_type': blob.content_type,
                    'metadata': blob.metadata or {}
                }
                files.append(file_info)
            
            logger.info(f"Listed {len(files)} files with prefix '{prefix}'")
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from Cloud Storage.
        
        Args:
            file_path: Path of the file to delete.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            blob = self.bucket.blob(file_path)
            blob.delete()
            
            logger.info(f"File deleted: gs://{self.bucket_name}/{file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return False
    
    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists in Cloud Storage.
        
        Args:
            file_path: Path of the file to check.
            
        Returns:
            True if file exists, False otherwise.
        """
        try:
            blob = self.bucket.blob(file_path)
            return blob.exists()
            
        except Exception as e:
            logger.error(f"Failed to check file existence {file_path}: {e}")
            return False
    
    def get_file_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a file in Cloud Storage.
        
        Args:
            file_path: Path of the file.
            
        Returns:
            File metadata dictionary or None if file doesn't exist.
        """
        try:
            blob = self.bucket.blob(file_path)
            blob.reload()
            
            return {
                'name': blob.name,
                'size': blob.size,
                'created': blob.time_created.isoformat() if blob.time_created else None,
                'updated': blob.updated.isoformat() if blob.updated else None,
                'content_type': blob.content_type,
                'etag': blob.etag,
                'metadata': blob.metadata or {}
            }
            
        except Exception as e:
            logger.error(f"Failed to get file metadata {file_path}: {e}")
            return None
    
    def create_versioned_path(
        self,
        base_path: str,
        version: Optional[str] = None
    ) -> str:
        """
        Create a versioned path for storing files.
        
        Args:
            base_path: Base path for the file.
            version: Version string (uses timestamp if None).
            
        Returns:
            Versioned path string.
        """
        if version is None:
            version = get_timestamp()
        
        # Remove file extension if present
        base_path_obj = Path(base_path)
        if base_path_obj.suffix:
            versioned_path = f"{base_path_obj.parent}/{base_path_obj.stem}_{version}{base_path_obj.suffix}"
        else:
            versioned_path = f"{base_path}_{version}"
        
        return versioned_path.replace('\\', '/')
    
    def sync_directory(
        self,
        local_directory: str,
        remote_prefix: str,
        direction: str = 'upload'
    ) -> Dict[str, Any]:
        """
        Sync a local directory with Cloud Storage.
        
        Args:
            local_directory: Local directory path.
            remote_prefix: Remote prefix in the bucket.
            direction: Sync direction ('upload', 'download', 'both').
            
        Returns:
            Dictionary with sync results.
        """
        results = {
            'uploaded': {},
            'downloaded': {},
            'skipped': {},
            'errors': []
        }
        
        if direction in ['upload', 'both']:
            # Upload local files that are newer or don't exist remotely
            upload_results = self.upload_directory(local_directory, remote_prefix)
            results['uploaded'] = upload_results
        
        if direction in ['download', 'both']:
            # Download remote files that are newer or don't exist locally
            download_results = self.download_directory(remote_prefix, local_directory)
            results['downloaded'] = download_results
        
        return results

class ArtifactManager:
    """Artifact management utility for MLOps pipeline."""
    
    def __init__(self, storage_manager: Optional[CloudStorageManager] = None):
        self.storage = storage_manager or CloudStorageManager()
        self.config = get_config()
        
    def save_model_artifact(
        self,
        model_path: str,
        model_name: str,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a model artifact to Cloud Storage.
        
        Args:
            model_path: Local path to the model file.
            model_name: Name of the model.
            version: Model version (uses timestamp if None).
            metadata: Additional metadata to store.
            
        Returns:
            Remote path where the model was saved.
        """
        if version is None:
            version = get_timestamp()
        
        # Create remote path
        remote_path = f"{self.config.storage.models_path}{model_name}/{version}/model.joblib"
        
        # Prepare metadata
        artifact_metadata = {
            'model_name': model_name,
            'version': version,
            'upload_time': get_timestamp(),
            'local_path': model_path
        }
        
        if metadata:
            artifact_metadata.update(metadata)
        
        # Upload model
        success = self.storage.upload_file(
            model_path,
            remote_path,
            metadata={k: str(v) for k, v in artifact_metadata.items()}
        )
        
        if success:
            logger.info(f"Model artifact saved: {model_name} v{version}")
            return remote_path
        else:
            raise Exception(f"Failed to save model artifact: {model_name}")
    
    def load_model_artifact(
        self,
        model_name: str,
        version: str = "latest",
        local_path: Optional[str] = None
    ) -> str:
        """
        Load a model artifact from Cloud Storage.
        
        Args:
            model_name: Name of the model.
            version: Model version ("latest" for most recent).
            local_path: Local path to download to (uses temp file if None).
            
        Returns:
            Local path where the model was downloaded.
        """
        if version == "latest":
            # Find the latest version
            prefix = f"{self.config.storage.models_path}{model_name}/"
            files = self.storage.list_files(prefix)
            
            if not files:
                raise Exception(f"No model artifacts found for {model_name}")
            
            # Sort by creation time and get the latest
            files.sort(key=lambda x: x['created'], reverse=True)
            latest_file = files[0]
            remote_path = latest_file['name']
        else:
            remote_path = f"{self.config.storage.models_path}{model_name}/{version}/model.joblib"
        
        # Determine local path
        if local_path is None:
            temp_dir = tempfile.mkdtemp()
            local_path = os.path.join(temp_dir, f"{model_name}_{version}.joblib")
        
        # Download model
        success = self.storage.download_file(remote_path, local_path)
        
        if success:
            logger.info(f"Model artifact loaded: {model_name} v{version}")
            return local_path
        else:
            raise Exception(f"Failed to load model artifact: {model_name} v{version}")
    
    def list_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """
        List all versions of a model.
        
        Args:
            model_name: Name of the model.
            
        Returns:
            List of model version information.
        """
        prefix = f"{self.config.storage.models_path}{model_name}/"
        files = self.storage.list_files(prefix)
        
        versions = []
        for file_info in files:
            if file_info['name'].endswith('/model.joblib'):
                # Extract version from path
                path_parts = file_info['name'].split('/')
                if len(path_parts) >= 3:
                    version = path_parts[-2]
                    versions.append({
                        'version': version,
                        'created': file_info['created'],
                        'size': file_info['size'],
                        'metadata': file_info['metadata']
                    })
        
        # Sort by creation time
        versions.sort(key=lambda x: x['created'], reverse=True)
        
        return versions
