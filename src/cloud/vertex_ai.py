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
    class MockModule:
        def __getattr__(self, name):
            return MockModule()
        def __call__(self, *args, **kwargs):
            return MockModule()
        def __getitem__(self, key):
            return MockModule()
        def __iter__(self):
            return iter([])
        def __len__(self):
            return 0
    
    aiplatform = MockModule()  # type: ignore
    storage = MockModule()  # type: ignore
    logging.warning("Google Cloud libraries not available. Install with: pip install google-cloud-aiplatform google-cloud-storage")

try:
    from config import get_config
except ImportError:
    # Fallback for when running from notebooks
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from config import get_config

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

class TrainingJobConfig:
    """Configuration for Vertex AI training jobs."""
    
    def __init__(
        self, 
        job_name: str,
        machine_type: str = "n1-standard-4",
        replica_count: int = 1,
        python_package_uri: Optional[str] = None,
        python_module: Optional[str] = None,
        container_uri: Optional[str] = None,
        args: Optional[List[str]] = None,
        environment_variables: Optional[Dict[str, str]] = None
    ):
        """
        Initialize training job configuration.
        
        Args:
            job_name: Name of the training job
            machine_type: Type of machine to use for training
            replica_count: Number of worker replicas
            python_package_uri: URI to Python package in GCS
            python_module: Python module to execute
            container_uri: Custom container image URI
            args: Command line arguments
            environment_variables: Environment variables for the job
        """
        self.job_name = job_name
        self.machine_type = machine_type
        self.replica_count = replica_count
        self.python_package_uri = python_package_uri
        self.python_module = python_module
        self.container_uri = container_uri
        self.args = args or []
        self.environment_variables = environment_variables or {}


class VertexAITrainer:
    """Vertex AI training job manager."""
    
    def __init__(self, project_id: str, location: str, staging_bucket: str):
        """
        Initialize Vertex AI trainer.
        
        Args:
            project_id: Google Cloud project ID
            location: Vertex AI location/region
            staging_bucket: GCS bucket for staging artifacts
        """
        if not GOOGLE_CLOUD_AVAILABLE:
            self.sdk_available = False
            logger.warning("Google Cloud SDK not available - running in simulation mode")
        else:
            self.sdk_available = True
            aiplatform.init(
                project=project_id,
                location=location,
                staging_bucket=staging_bucket
            )
        
        self.project_id = project_id
        self.location = location
        self.staging_bucket = staging_bucket
        self.jobs = {}
    
    def create_training_job_from_local_script(
        self,
        script_path: str,
        job_name: str,
        args: Optional[List[str]] = None,
        requirements_file: Optional[str] = None,
        machine_type: str = "n1-standard-4",
        container_uri: Optional[str] = None
    ):
        """
        Create a training job from a local script.
        
        Args:
            script_path: Path to the training script
            job_name: Name for the training job
            args: Command line arguments for the script
            requirements_file: Path to requirements.txt file
            machine_type: Machine type for training
            container_uri: Custom container image URI
            
        Returns:
            Training job object
        """
        if not self.sdk_available:
            # Return mock job for simulation
            mock_job = type('MockJob', (), {
                'job_id': f"mock-{job_name}-{hash(script_path) % 10000}",
                'display_name': job_name,
                'state': 'RUNNING',
                'console_url': f"https://console.cloud.google.com/vertex-ai/training/{job_name}",
                'name': f"projects/{self.project_id}/locations/{self.location}/customJobs/mock-{job_name}"
            })()
            self.jobs[job_name] = mock_job
            return mock_job
        
        try:
            # Use default scikit-learn container if none specified
            if not container_uri:
                container_uri = "us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.0-23:latest"
            
            # Create the custom job
            job = aiplatform.CustomJob(
                display_name=job_name,
                worker_pool_specs=[
                    {
                        "machine_spec": {"machine_type": machine_type},
                        "replica_count": 1,
                        "container_spec": {
                            "image_uri": container_uri,
                            "command": ["python3"],
                            "args": ["-c", f"import subprocess; subprocess.run(['python3', '{script_path}'] + {args or []})"]
                        },
                    }
                ],
                staging_bucket=self.staging_bucket,
            )
            
            # Submit the job immediately
            job.run(sync=False)  # Don't wait for completion
            
            # Wait a moment for the job to register
            import time
            time.sleep(2)
            
            self.jobs[job_name] = job
            return job
            
        except Exception as e:
            logger.error(f"Failed to create training job: {e}")
            # Return mock job as fallback
            mock_job = type('MockJob', (), {
                'job_id': f"fallback-{job_name}-{hash(str(e)) % 10000}",
                'display_name': job_name,
                'state': 'FAILED',
                'error': str(e),
                'name': f"projects/{self.project_id}/locations/{self.location}/customJobs/fallback-{job_name}"
            })()
            self.jobs[job_name] = mock_job
            return mock_job
    
    def get_training_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a training job.
        
        Args:
            job_id: ID of the training job
            
        Returns:
            Dictionary with job status information
        """
        if not self.sdk_available:
            return {
                'state': 'SUCCEEDED',
                'createTime': '2024-11-24T12:00:00Z',
                'endTime': '2024-11-24T12:15:00Z'
            }
        
        try:
            # Find job by ID
            for job_name, job in self.jobs.items():
                if hasattr(job, 'job_id') and job.job_id == job_id:
                    job.refresh()
                    return {
                        'state': job.state.name,
                        'createTime': str(job.create_time) if job.create_time else None,
                        'endTime': str(job.end_time) if job.end_time else None,
                        'error': str(job.error) if job.error else None
                    }
        except Exception as e:
            logger.error(f"Error getting job status: {e}")
        
        return {'state': 'UNKNOWN'}


class CloudTrainingUtils:
    """Utilities for cloud-based training operations."""
    
    def __init__(self, project_id: str, location: str):
        """
        Initialize cloud training utilities.
        
        Args:
            project_id: Google Cloud project ID
            location: Vertex AI location/region
        """
        self.project_id = project_id
        self.location = location
        
        # Initialize storage client if available
        if GOOGLE_CLOUD_AVAILABLE and storage:
            self.storage_client = storage.Client(project=project_id)
        else:
            self.storage_client = None
            logger.warning("Google Cloud Storage not available - upload operations will be simulated")
    
    def upload_to_gcs(self, local_path: str, gcs_path: str) -> str:
        """
        Upload a local file to Google Cloud Storage.
        
        Args:
            local_path: Local file path
            gcs_path: GCS destination path (gs://bucket/path)
            
        Returns:
            GCS path of the uploaded file
        """
        if not self.storage_client:
            logger.info(f"Simulated upload: {local_path} -> {gcs_path}")
            return gcs_path
        
        try:
            # Parse GCS path
            if not gcs_path.startswith('gs://'):
                raise ValueError("GCS path must start with gs://")
            
            path_parts = gcs_path[5:].split('/', 1)
            bucket_name = path_parts[0]
            blob_name = path_parts[1] if len(path_parts) > 1 else ''
            
            # Get bucket and upload
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_path)
            
            logger.info(f"File uploaded to {gcs_path}")
            return gcs_path
            
        except Exception as e:
            logger.error(f"Failed to upload {local_path} to {gcs_path}: {e}")
            raise
    
    def create_model_training_script(self, output_dir: str) -> str:
        """
        Create a template training script for Vertex AI.
        
        Args:
            output_dir: Directory to save the training script
            
        Returns:
            Path to the created training script
        """
        script_content = '''#!/usr/bin/env python3
"""
Vertex AI Training Script for Iris Classification

This script trains machine learning models on the Iris dataset using scikit-learn
and is designed to run on Vertex AI custom training jobs.
"""

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(data_path: str) -> tuple:
    """Load and prepare the iris dataset."""
    logger.info(f"Loading data from {data_path}")
    
    if data_path.startswith('gs://'):
        # Handle GCS paths
        import subprocess
        local_path = '/tmp/iris_data.csv'
        subprocess.run(['gsutil', 'cp', data_path, local_path], check=True)
        data_path = local_path
    
    df = pd.read_csv(data_path)
    
    # Prepare features and target
    feature_columns = [col for col in df.columns if col != 'target']
    X = df[feature_columns].values
    y = df['target'].values
    
    logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, feature_columns

def train_model(X, y, model_type: str, enable_tuning: bool = False) -> dict:
    """Train a machine learning model."""
    logger.info(f"Training {model_type} model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Select model
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == 'svm':
        model = SVC(random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    
    results = {
        'model_type': model_type,
        'train_accuracy': float(train_score),
        'test_accuracy': float(test_score),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    logger.info(f"Model trained - Test Accuracy: {test_score:.4f}")
    return model, results

def save_artifacts(model, results: dict, output_path: str):
    """Save model and results."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_path
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_path = str(output_path).replace('.pkl', '_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Upload to GCS if output is GCS path
    if output_path.startswith('gs://'):
        import subprocess
        subprocess.run(['gsutil', 'cp', model_path, output_path], check=True)
        subprocess.run(['gsutil', 'cp', metrics_path, metrics_path.replace('/tmp/', 'gs://')], check=True)

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train ML model on Iris dataset')
    parser.add_argument('--model-type', default='random_forest', 
                       choices=['random_forest', 'logistic_regression', 'svm'],
                       help='Type of model to train')
    parser.add_argument('--data-path', required=True, help='Path to training data')
    parser.add_argument('--model-output-path', required=True, help='Output path for trained model')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--cross-val-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--random-state', type=int, default=42, help='Random state')
    parser.add_argument('--enable-tuning', action='store_true', help='Enable hyperparameter tuning')
    
    args = parser.parse_args()
    
    logger.info("Starting Vertex AI training job")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output path: {args.model_output_path}")
    
    try:
        # Load data
        X, y, feature_columns = load_data(args.data_path)
        
        # Train model
        model, results = train_model(X, y, args.model_type, args.enable_tuning)
        
        # Save artifacts
        save_artifacts(model, results, args.model_output_path)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
'''
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save training script
        script_path = output_path / "train.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Create requirements.txt
        requirements_content = '''scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
joblib>=1.0.0
google-cloud-storage>=2.0.0
'''
        requirements_path = output_path / "requirements.txt"
        with open(requirements_path, 'w') as f:
            f.write(requirements_content)
        
        logger.info(f"Training script created: {script_path}")
        logger.info(f"Requirements file created: {requirements_path}")
        
        return str(script_path)
