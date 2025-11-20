"""
Vertex AI integration utilities for MLOps pipeline.

This module provides integration with Google Cloud Vertex AI including:
- Training job management
- Model deployment
- Endpoint creation and management
- Prediction services
"""

import logging
import os
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

# Handle optional Google Cloud dependencies
try:
    from google.cloud import aiplatform  # type: ignore
    from google.cloud import storage  # type: ignore
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False
    # Create mock modules for type checking
    aiplatform = None  # type: ignore
    storage = None  # type: ignore
    logging.warning("Google Cloud libraries not available. Install with: pip install google-cloud-aiplatform google-cloud-storage")

from ..config import get_config

logger = logging.getLogger(__name__)

class VertexAIManager:
    """Vertex AI operations manager."""
    
    def __init__(self, project_id: Optional[str] = None, region: Optional[str] = None):
        if not GOOGLE_CLOUD_AVAILABLE or aiplatform is None:
            raise ImportError("Google Cloud libraries not installed. Run: pip install google-cloud-aiplatform google-cloud-storage")
        
        config = get_config()
        
        self.project_id = project_id or config.gcp.project_id
        self.region = region or config.gcp.region
        self.bucket_name = config.storage.bucket_name
        
        # Initialize Vertex AI
        aiplatform.init(  # type: ignore
            project=self.project_id,
            location=self.region,
            staging_bucket=f"gs://{self.bucket_name}"
        )
        
        logger.info(f"Vertex AI initialized for project {self.project_id} in {self.region}")
    
    def create_custom_training_job(
        self,
        display_name: str,
        container_uri: str,
        model_serving_container_uri: str,
        args: Optional[List[str]] = None,
        machine_type: str = "n1-standard-4",
        replica_count: int = 1
    ) -> Dict[str, Any]:
        """
        Create a custom training job.
        
        Args:
            display_name: Display name for the training job.
            container_uri: URI of the training container.
            model_serving_container_uri: URI of the model serving container.
            args: Arguments for the training script.
            machine_type: Machine type for training.
            replica_count: Number of replicas.
            
        Returns:
            Dictionary containing job information.
        """
        logger.info(f"Creating custom training job: {display_name}")
        
        if aiplatform is None:
            raise ImportError("Google Cloud Vertex AI not available")
        
        job = aiplatform.CustomContainerTrainingJob(  # type: ignore
            display_name=display_name,
            container_uri=container_uri,
            command=args or []
        )
        
        # Submit the job
        model = job.run(
            machine_type=machine_type,
            replica_count=replica_count,
            sync=False  # Don't wait for completion
        )
        
        job_info = {
            'job_name': job.resource_name,
            'display_name': display_name,
            'state': job.state,
            'create_time': str(job.create_time),
            'model': model.resource_name if model else None
        }
        
        logger.info(f"Training job created: {job.resource_name}")
        return job_info
    
    def deploy_model(
        self,
        model_name: str,
        endpoint_name: str,
        machine_type: str = "n1-standard-2",
        min_replica_count: int = 1,
        max_replica_count: int = 1
    ) -> Dict[str, Any]:
        """
        Deploy a model to an endpoint.
        
        Args:
            model_name: Name of the model to deploy.
            endpoint_name: Name of the endpoint.
            machine_type: Machine type for the endpoint.
            min_replica_count: Minimum number of replicas.
            max_replica_count: Maximum number of replicas.
            
        Returns:
            Dictionary containing deployment information.
        """
        logger.info(f"Deploying model {model_name} to endpoint {endpoint_name}")
        
        # Get or create endpoint
        try:
            endpoint = aiplatform.Endpoint.list(
                filter=f'display_name="{endpoint_name}"'
            )[0]
            logger.info(f"Using existing endpoint: {endpoint.display_name}")
        except (IndexError, Exception):
            endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
            logger.info(f"Created new endpoint: {endpoint.display_name}")
        
        # Get model
        models = aiplatform.Model.list(filter=f'display_name="{model_name}"')
        if not models:
            raise ValueError(f"Model {model_name} not found")
        
        model = models[0]
        
        # Deploy model to endpoint
        deployed_model = endpoint.deploy(
            model=model,
            machine_type=machine_type,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count
        )
        
        deployment_info = {
            'endpoint_name': endpoint.display_name,
            'endpoint_id': endpoint.resource_name,
            'model_name': model.display_name,
            'model_id': model.resource_name,
            'deployed_model_id': deployed_model.id
        }
        
        logger.info(f"Model deployed successfully to endpoint {endpoint_name}")
        return deployment_info
    
    def predict(
        self,
        endpoint_name: str,
        instances: List[List[float]]
    ) -> List[Any]:
        """
        Make predictions using a deployed model.
        
        Args:
            endpoint_name: Name of the endpoint.
            instances: List of instances to predict.
            
        Returns:
            List of predictions.
        """
        # Get endpoint
        endpoints = aiplatform.Endpoint.list(
            filter=f'display_name="{endpoint_name}"'
        )
        if not endpoints:
            raise ValueError(f"Endpoint {endpoint_name} not found")
        
        endpoint = endpoints[0]
        
        # Make predictions
        predictions = endpoint.predict(instances=instances)
        
        return predictions.predictions
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all models in the project.
        
        Returns:
            List of model information dictionaries.
        """
        models = aiplatform.Model.list()
        
        model_list = []
        for model in models:
            model_info = {
                'name': model.display_name,
                'resource_name': model.resource_name,
                'create_time': str(model.create_time),
                'update_time': str(model.update_time)
            }
            model_list.append(model_info)
        
        return model_list
    
    def list_endpoints(self) -> List[Dict[str, Any]]:
        """
        List all endpoints in the project.
        
        Returns:
            List of endpoint information dictionaries.
        """
        endpoints = aiplatform.Endpoint.list()
        
        endpoint_list = []
        for endpoint in endpoints:
            endpoint_info = {
                'name': endpoint.display_name,
                'resource_name': endpoint.resource_name,
                'create_time': str(endpoint.create_time),
                'update_time': str(endpoint.update_time)
            }
            endpoint_list.append(endpoint_info)
        
        return endpoint_list

class CloudStorageManager:
    """Google Cloud Storage operations manager."""
    
    def __init__(self, bucket_name: Optional[str] = None):
        if not GOOGLE_CLOUD_AVAILABLE:
            raise ImportError("Google Cloud libraries not installed")
        
        config = get_config()
        self.bucket_name = bucket_name or config.storage.bucket_name
        
        # Initialize storage client
        self.client = storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)
        
        logger.info(f"Cloud Storage initialized for bucket: {self.bucket_name}")
    
    def upload_file(self, local_path: str, blob_name: str) -> str:
        """
        Upload a file to Cloud Storage.
        
        Args:
            local_path: Local file path.
            blob_name: Name of the blob in Cloud Storage.
            
        Returns:
            GCS URI of the uploaded file.
        """
        blob = self.bucket.blob(blob_name)
        blob.upload_from_filename(local_path)
        
        gcs_uri = f"gs://{self.bucket_name}/{blob_name}"
        logger.info(f"File uploaded to {gcs_uri}")
        
        return gcs_uri
    
    def download_file(self, blob_name: str, local_path: str) -> None:
        """
        Download a file from Cloud Storage.
        
        Args:
            blob_name: Name of the blob in Cloud Storage.
            local_path: Local path to save the file.
        """
        blob = self.bucket.blob(blob_name)
        
        # Create directory if it doesn't exist
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        
        blob.download_to_filename(local_path)
        logger.info(f"File downloaded from gs://{self.bucket_name}/{blob_name} to {local_path}")
    
    def list_blobs(self, prefix: Optional[str] = None) -> List[str]:
        """
        List blobs in the bucket.
        
        Args:
            prefix: Optional prefix to filter blobs.
            
        Returns:
            List of blob names.
        """
        blobs = self.bucket.list_blobs(prefix=prefix)
        return [blob.name for blob in blobs]
    
    def delete_blob(self, blob_name: str) -> None:
        """
        Delete a blob from Cloud Storage.
        
        Args:
            blob_name: Name of the blob to delete.
        """
        blob = self.bucket.blob(blob_name)
        blob.delete()
        logger.info(f"Blob deleted: gs://{self.bucket_name}/{blob_name}")

def verify_cloud_setup() -> Dict[str, bool]:
    """
    Verify that Google Cloud setup is working correctly.
    
    Returns:
        Dictionary containing verification results.
    """
    results = {
        'google_cloud_available': GOOGLE_CLOUD_AVAILABLE,
        'credentials_set': False,
        'vertex_ai_accessible': False,
        'storage_accessible': False
    }
    
    if not GOOGLE_CLOUD_AVAILABLE:
        logger.error("Google Cloud libraries not available")
        return results
    
    # Check credentials
    try:
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if credentials_path and Path(credentials_path).exists():
            results['credentials_set'] = True
        else:
            logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set or file not found")
    except Exception as e:
        logger.error(f"Error checking credentials: {e}")
    
    # Check Vertex AI access
    try:
        config = get_config()
        aiplatform.init(
            project=config.gcp.project_id,
            location=config.gcp.region
        )
        # Try to list models (this will fail if no access)
        aiplatform.Model.list()
        results['vertex_ai_accessible'] = True
    except Exception as e:
        logger.error(f"Vertex AI not accessible: {e}")
    
    # Check Cloud Storage access
    try:
        config = get_config()
        client = storage.Client()
        bucket = client.bucket(config.storage.bucket_name)
        # Try to list blobs (this will fail if no access)
        list(bucket.list_blobs(max_results=1))
        results['storage_accessible'] = True
    except Exception as e:
        logger.error(f"Cloud Storage not accessible: {e}")
    
    return results
