"""
Vertex AI Training Module

This module provides utilities for training machine learning models on Google Cloud
Vertex AI, including custom training jobs, hyperparameter tuning, and model registry
integration.

Classes:
    VertexAITrainer: Main class for Vertex AI training operations
    TrainingJobConfig: Configuration for Vertex AI training jobs
    CloudTrainingUtils: Utility functions for cloud training

Author: MLOps Team
Version: 1.0.0
"""

import os
import json
import logging
import subprocess
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field

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

# Check for Google Cloud Python SDK
VERTEX_AI_SDK_AVAILABLE = False
aiplatform_module = None
storage_module = None

try:
    import google.cloud.aiplatform as aiplatform_module
    import google.cloud.storage as storage_module
    VERTEX_AI_SDK_AVAILABLE = True
except ImportError:
    logging.warning("Google Cloud AI Platform SDK not available. "
                   "Install with: pip install google-cloud-aiplatform google-cloud-storage")


@dataclass
class TrainingJobConfig:
    """Configuration for Vertex AI training jobs."""
    display_name: str
    container_uri: str
    model_serving_container_uri: Optional[str] = None
    machine_type: str = "n1-standard-4"
    replica_count: int = 1
    args: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    python_package_gcs_uri: Optional[str] = None
    python_module: Optional[str] = None
    requirements: List[str] = field(default_factory=list)
    base_output_dir: Optional[str] = None
    enable_web_access: bool = False
    timeout: str = "7200s"  # 2 hours
    restart_job_on_worker_restart: bool = False


@dataclass
class CloudTrainingJob:
    """Represents a cloud training job."""
    job_id: str
    display_name: str
    state: str
    create_time: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    training_output_dir: Optional[str] = None
    console_url: Optional[str] = None


class CloudTrainingUtils:
    """
    Utilities for cloud-based model training without direct SDK dependencies.
    
    This class provides cloud training capabilities using gcloud CLI commands,
    making it more robust for different environments.
    """
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        """
        Initialize cloud training utilities.
        
        Args:
            project_id: Google Cloud project ID
            location: Vertex AI location/region
        """
        self.project_id = project_id
        self.location = location
        
        if not GCLOUD_AVAILABLE:
            raise RuntimeError("Google Cloud CLI (gcloud) is not available. "
                             "Please install and configure it.")
        
        # Set project
        self._run_gcloud_command(['config', 'set', 'project', project_id])
        
        logger.info(f"CloudTrainingUtils initialized - Project: {project_id}, Location: {location}")
    
    def _run_gcloud_command(self, cmd: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
        """
        Run a gcloud command.
        
        Args:
            cmd: Command arguments
            capture_output: Whether to capture output
            
        Returns:
            Completed process
        """
        full_cmd = ['gcloud'] + cmd
        logger.debug(f"Running: {' '.join(full_cmd)}")
        
        result = subprocess.run(
            full_cmd,
            capture_output=capture_output,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode != 0:
            logger.error(f"Command failed: {' '.join(full_cmd)}")
            logger.error(f"Error output: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, full_cmd, result.stdout, result.stderr)
        
        return result
    
    def create_training_package(self, source_dir: str, output_path: str) -> str:
        """
        Create a Python training package.
        
        Args:
            source_dir: Directory containing training code
            output_path: Output path for the package
            
        Returns:
            Path to created package
        """
        import tarfile
        
        logger.info(f"Creating training package: {source_dir} -> {output_path}")
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create tar.gz package
        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(source_dir, arcname=".")
        
        logger.info(f"Training package created: {output_path}")
        return output_path
    
    def upload_to_gcs(self, local_path: str, gcs_path: str) -> str:
        """
        Upload a file to Google Cloud Storage.
        
        Args:
            local_path: Local file path
            gcs_path: GCS destination path (gs://bucket/path)
            
        Returns:
            GCS path
        """
        logger.info(f"Uploading to GCS: {local_path} -> {gcs_path}")
        
        self._run_gcloud_command(['storage', 'cp', local_path, gcs_path])
        
        logger.info(f"Upload completed: {gcs_path}")
        return gcs_path
    
    def submit_custom_job(self, config: TrainingJobConfig) -> CloudTrainingJob:
        """
        Submit a custom training job using gcloud.
        
        Args:
            config: Training job configuration
            
        Returns:
            CloudTrainingJob instance
        """
        logger.info(f"Submitting custom training job: {config.display_name}")
        
        # Build gcloud command
        cmd = [
            'ai', 'custom-jobs', 'create',
            '--region', self.location,
            '--display-name', config.display_name,
        ]
        
        # Create job spec
        job_spec = {
            "workerPoolSpecs": [{
                "machineSpec": {
                    "machineType": config.machine_type
                },
                "replicaCount": config.replica_count,
                "containerSpec": {
                    "imageUri": config.container_uri,
                    "args": config.args
                }
            }]
        }
        
        # Add environment variables
        if config.environment_variables:
            env_vars = [{"name": k, "value": v} for k, v in config.environment_variables.items()]
            job_spec["workerPoolSpecs"][0]["containerSpec"]["env"] = env_vars
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(job_spec, f, indent=2)
            config_file = f.name
        
        try:
            cmd.extend(['--config', config_file])
            
            # Submit job
            result = self._run_gcloud_command(cmd)
            
            # Parse job ID from output
            output_lines = result.stdout.strip().split('\n')
            job_id = None
            for line in output_lines:
                if 'name:' in line:
                    job_id = line.split('name:')[-1].strip()
                    break
            
            if not job_id:
                # Fallback: extract from last line
                job_id = output_lines[-1].strip()
            
            # Create console URL
            console_url = (f"https://console.cloud.google.com/vertex-ai/locations/{self.location}/"
                          f"training/{job_id}?project={self.project_id}")
            
            training_job = CloudTrainingJob(
                job_id=job_id,
                display_name=config.display_name,
                state="SUBMITTED",
                training_output_dir=config.base_output_dir,
                console_url=console_url
            )
            
            logger.info(f"Training job submitted - Job ID: {job_id}")
            logger.info(f"Monitor at: {console_url}")
            
            return training_job
            
        finally:
            # Clean up temporary file
            os.unlink(config_file)
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a training job.
        
        Args:
            job_id: Job ID or full resource name
            
        Returns:
            Job status information
        """
        cmd = [
            'ai', 'custom-jobs', 'describe', job_id,
            '--region', self.location,
            '--format', 'json'
        ]
        
        result = self._run_gcloud_command(cmd)
        job_info = json.loads(result.stdout)
        
        return {
            "name": job_info.get("name", ""),
            "displayName": job_info.get("displayName", ""),
            "state": job_info.get("state", "UNKNOWN"),
            "createTime": job_info.get("createTime"),
            "startTime": job_info.get("startTime"),
            "endTime": job_info.get("endTime"),
            "error": job_info.get("error")
        }
    
    def list_training_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List recent training jobs.
        
        Args:
            limit: Maximum number of jobs to return
            
        Returns:
            List of job information
        """
        cmd = [
            'ai', 'custom-jobs', 'list',
            '--region', self.location,
            '--limit', str(limit),
            '--format', 'json'
        ]
        
        result = self._run_gcloud_command(cmd)
        jobs = json.loads(result.stdout) if result.stdout.strip() else []
        
        return jobs
    
    def create_model_training_script(self, output_dir: str) -> str:
        """
        Create a template training script for Vertex AI.
        
        Args:
            output_dir: Directory to create the training script
            
        Returns:
            Path to the created training script
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create main training script
        training_script = output_path / "train.py"
        training_content = '''#!/usr/bin/env python3
"""
Vertex AI Training Script

This script demonstrates model training on Vertex AI using the MLOps pipeline.
"""

import argparse
import os
import sys
import json
import logging
import joblib
from pathlib import Path

# Add project source to path
sys.path.append('/app/src')

# Import our training modules
from models.trainer import ModelTrainer, ModelConfig
from data.data_loader import DataLoader
from data.preprocessor import DataPreprocessor
from config import Config

def setup_logging():
    """Setup logging for training."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Vertex AI Model Training')
    
    # Model parameters
    parser.add_argument('--model-type', default='random_forest',
                       help='Type of model to train')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility')
    
    # Data parameters
    parser.add_argument('--data-path', required=True,
                       help='Path to training data (GCS or local)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size ratio')
    
    # Training parameters
    parser.add_argument('--cross-val-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--enable-tuning', action='store_true',
                       help='Enable hyperparameter tuning')
    
    # Output parameters
    parser.add_argument('--model-dir', default='/tmp/model',
                       help='Directory to save trained model')
    parser.add_argument('--output-dir', 
                       default=os.environ.get('AIP_MODEL_DIR', '/tmp/outputs'),
                       help='Directory for training outputs')
    
    return parser.parse_args()

def main():
    """Main training function."""
    logger = setup_logging()
    args = parse_args()
    
    logger.info("Starting Vertex AI training job")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Load data
        logger.info(f"Loading data from: {args.data_path}")
        data_loader = DataLoader()
        
        if args.data_path.endswith('.csv'):
            data = data_loader.load_from_csv(args.data_path)
        else:
            data = data_loader.load_iris_dataset()
        
        # Preprocess data
        logger.info("Preprocessing data")
        preprocessor = DataPreprocessor(random_state=args.random_state)
        X_train, X_test, y_train, y_test = preprocessor.create_train_test_split(
            data, test_size=args.test_size
        )
        
        # Train model
        logger.info(f"Training {args.model_type} model")
        trainer = ModelTrainer(random_state=args.random_state)
        trainer.load_data(X_train, X_test, y_train, y_test)
        
        # Get model configuration
        models = trainer.get_default_models()
        if args.model_type in models:
            config = models[args.model_type]
            config.enable_hyperparameter_tuning = args.enable_tuning
            config.cross_validation_folds = args.cross_val_folds
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
        
        # Train the model
        result = trainer.train_model(config)
        
        # Save model and results
        model_dir = Path(args.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save trained model
        model_path = model_dir / "model.pkl"
        trainer.save_model(config.name, model_path)
        
        # Save results
        results_path = output_dir / "training_results.json"
        results_data = {
            'model_name': result.model_name,
            'algorithm': result.algorithm,
            'test_accuracy': float(result.test_accuracy),
            'cross_val_mean': float(result.cross_val_mean),
            'cross_val_std': float(result.cross_val_std),
            'training_time': float(result.training_time),
            'best_parameters': result.best_parameters
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save metrics for Vertex AI
        metrics_path = output_dir / "metrics.json"
        metrics = {
            'accuracy': float(result.test_accuracy),
            'precision': float(result.precision),
            'recall': float(result.recall),
            'f1_score': float(result.f1_score)
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Test Accuracy: {result.test_accuracy:.4f}")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Results saved to: {results_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
'''
        
        training_script.write_text(training_content)
        
        # Create setup.py
        setup_py = output_path / "setup.py"
        setup_content = '''
from setuptools import setup, find_packages

setup(
    name="vertex-ai-training",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "google-cloud-aiplatform>=1.35.0",
        "google-cloud-storage>=2.10.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "joblib>=1.3.0",
        "pyyaml>=6.0"
    ],
    python_requires=">=3.8"
)
'''
        setup_py.write_text(setup_content.strip())
        
        # Create Dockerfile
        dockerfile = output_path / "Dockerfile"
        dockerfile_content = '''
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY setup.py .
RUN pip install --no-cache-dir -e .

# Copy source code
COPY . .

# Set the entrypoint
ENTRYPOINT ["python", "train.py"]
'''
        dockerfile.write_text(dockerfile_content.strip())
        
        logger.info(f"Training script template created in: {output_dir}")
        return str(training_script)
    
    def build_and_push_container(self, source_dir: str, 
                                image_name: str, 
                                tag: str = "latest") -> str:
        """
        Build and push a container image to Google Container Registry.
        
        Args:
            source_dir: Directory containing Dockerfile and source code
            image_name: Name for the container image
            tag: Tag for the image
            
        Returns:
            Full container URI
        """
        # Construct full image URI
        image_uri = f"gcr.io/{self.project_id}/{image_name}:{tag}"
        
        logger.info(f"Building container image: {image_uri}")
        
        # Build image
        build_cmd = [
            'builds', 'submit',
            '--tag', image_uri,
            source_dir
        ]
        
        self._run_gcloud_command(build_cmd)
        
        logger.info(f"Container image built and pushed: {image_uri}")
        return image_uri


class VertexAITrainer:
    """
    High-level interface for Vertex AI training operations.
    
    Provides both SDK-based and CLI-based training capabilities.
    """
    
    def __init__(self, project_id: str, location: str = "us-central1",
                 staging_bucket: Optional[str] = None):
        """
        Initialize Vertex AI trainer.
        
        Args:
            project_id: Google Cloud project ID
            location: Vertex AI location/region
            staging_bucket: GCS bucket for staging artifacts
        """
        self.project_id = project_id
        self.location = location
        self.staging_bucket = staging_bucket
        
        # Initialize cloud utilities
        self.cloud_utils = CloudTrainingUtils(project_id, location)
        
        # Try to initialize SDK if available
        self.sdk_available = VERTEX_AI_SDK_AVAILABLE
        self.storage_client = None
        
        if self.sdk_available and aiplatform_module and storage_module:
            try:
                aiplatform_module.init(
                    project=project_id,
                    location=location,
                    staging_bucket=staging_bucket
                )
                self.storage_client = storage_module.Client(project=project_id)
                logger.info("Vertex AI SDK initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Vertex AI SDK: {e}")
                self.sdk_available = False
        
        logger.info(f"VertexAITrainer initialized - SDK Available: {self.sdk_available}")
    
    def create_training_job_from_local_script(self, 
                                            script_path: str,
                                            job_name: str,
                                            args: List[str],
                                            requirements_file: Optional[str] = None,
                                            machine_type: str = "n1-standard-4") -> CloudTrainingJob:
        """
        Create and submit a training job from a local Python script.
        
        Args:
            script_path: Path to the training script
            job_name: Name for the training job
            args: Arguments to pass to the script
            requirements_file: Path to requirements.txt
            machine_type: Machine type for training
            
        Returns:
            CloudTrainingJob instance
        """
        logger.info(f"Creating training job from script: {script_path}")
        
        # Create temporary directory for packaging
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy script and requirements
            package_dir = Path(temp_dir) / "training_package"
            package_dir.mkdir()
            
            # Copy main script
            script_name = Path(script_path).name
            shutil.copy2(script_path, package_dir / script_name)
            
            # Copy requirements if provided
            if requirements_file and Path(requirements_file).exists():
                shutil.copy2(requirements_file, package_dir / "requirements.txt")
            
            # Create simple Dockerfile
            dockerfile_content = f'''
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY {script_name} .
ENTRYPOINT ["python", "{script_name}"]
'''
            (package_dir / "Dockerfile").write_text(dockerfile_content.strip())
            
            # Build and push container
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            image_name = f"{job_name}-{timestamp}"
            container_uri = self.cloud_utils.build_and_push_container(
                str(package_dir), image_name
            )
            
            # Create training job config
            config = TrainingJobConfig(
                display_name=f"{job_name}-{timestamp}",
                container_uri=container_uri,
                machine_type=machine_type,
                args=args,
                base_output_dir=f"gs://{self.project_id}-vertex-ai/training/{job_name}-{timestamp}"
            )
            
            # Submit job
            return self.cloud_utils.submit_custom_job(config)
    
    def get_training_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of a training job."""
        return self.cloud_utils.get_job_status(job_id)
    
    def list_training_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent training jobs."""
        return self.cloud_utils.list_training_jobs(limit)


def create_default_training_config(project_id: str, 
                                 job_name: str,
                                 container_uri: str,
                                 machine_type: str = "n1-standard-4") -> TrainingJobConfig:
    """
    Create a default training job configuration.
    
    Args:
        project_id: Google Cloud project ID
        job_name: Name for the training job
        container_uri: Docker container URI
        machine_type: Machine type for training
        
    Returns:
        Default TrainingJobConfig
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    display_name = f"{job_name}-{timestamp}"
    
    return TrainingJobConfig(
        display_name=display_name,
        container_uri=container_uri,
        machine_type=machine_type,
        replica_count=1,
        args=[],
        environment_variables={
            "PROJECT_ID": project_id,
            "TIMESTAMP": timestamp
        },
        base_output_dir=f"gs://{project_id}-vertex-ai/training_outputs/{display_name}",
        enable_web_access=False,
        timeout="7200s"
    )
