"""
Distributed Training Module

This module provides capabilities for distributed training using Vertex AI
and multi-node training configurations for scalable ML workflows.

Classes:
    DistributedTrainingConfig: Configuration for distributed training
    DistributedTrainer: Manager for distributed training operations
    MultiNodeConfig: Configuration for multi-node distributed training

Author: MLOps Team
Version: 1.0.0
"""

import os
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

# Local imports
from ..config import Config
from ..utils import setup_logging
from .vertex_ai import CloudTrainingUtils, TrainingJobConfig

logger = logging.getLogger(__name__)


class DistributionStrategy(Enum):
    """Supported distribution strategies."""
    SINGLE_NODE = "single_node"
    MULTI_NODE_GPU = "multi_node_gpu"
    MULTI_NODE_TPU = "multi_node_tpu"
    PARAMETER_SERVER = "parameter_server"
    MIRRORED_STRATEGY = "mirrored_strategy"


@dataclass
class WorkerPoolSpec:
    """Specification for a worker pool in distributed training."""
    replica_count: int
    machine_type: str
    accelerator_type: Optional[str] = None
    accelerator_count: int = 0
    disk_type: str = "pd-ssd"
    disk_size_gb: int = 100
    container_spec: Optional[Dict[str, Any]] = None
    python_package_spec: Optional[Dict[str, Any]] = None


@dataclass
class DistributedTrainingConfig:
    """Configuration for distributed training jobs."""
    display_name: str
    distribution_strategy: DistributionStrategy
    worker_pool_specs: List[WorkerPoolSpec]
    base_output_dir: str
    training_args: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    service_account: Optional[str] = None
    network: Optional[str] = None
    timeout: str = "86400s"  # 24 hours
    enable_web_access: bool = False
    scheduling: Optional[Dict[str, Any]] = None


@dataclass
class TrainingJobStatus:
    """Status information for a distributed training job."""
    job_id: str
    display_name: str
    state: str
    create_time: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    error_message: Optional[str] = None
    training_output_dir: Optional[str] = None
    console_url: Optional[str] = None


class DistributedTrainer:
    """
    Manager for distributed training operations on Vertex AI.
    
    Supports various distributed training strategies including multi-node GPU,
    TPU training, parameter server, and mirrored strategies.
    """
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        """
        Initialize distributed trainer.
        
        Args:
            project_id: Google Cloud project ID
            location: Vertex AI location/region
        """
        self.project_id = project_id
        self.location = location
        
        # Initialize cloud training utilities
        try:
            self.cloud_utils = CloudTrainingUtils(project_id, location)
            logger.info(f"Distributed trainer initialized for {project_id} in {location}")
        except Exception as e:
            logger.error(f"Failed to initialize distributed trainer: {e}")
            self.cloud_utils = None
    
    def create_single_node_config(self, display_name: str, 
                                 container_uri: str,
                                 machine_type: str = "n1-standard-4",
                                 accelerator_type: Optional[str] = None,
                                 accelerator_count: int = 0) -> DistributedTrainingConfig:
        """
        Create configuration for single-node training.
        
        Args:
            display_name: Display name for the training job
            container_uri: Container image URI
            machine_type: Machine type for training
            accelerator_type: GPU/TPU accelerator type
            accelerator_count: Number of accelerators
            
        Returns:
            DistributedTrainingConfig object
        """
        worker_pool = WorkerPoolSpec(
            replica_count=1,
            machine_type=machine_type,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
            container_spec={
                "image_uri": container_uri
            }
        )
        
        config = DistributedTrainingConfig(
            display_name=display_name,
            distribution_strategy=DistributionStrategy.SINGLE_NODE,
            worker_pool_specs=[worker_pool],
            base_output_dir=f"gs://{self.project_id}-vertex-ai/distributed-training/{display_name}"
        )
        
        logger.info(f"Created single-node config: {display_name}")
        return config
    
    def create_multi_gpu_config(self, display_name: str,
                               container_uri: str,
                               replica_count: int = 2,
                               machine_type: str = "n1-standard-8",
                               gpu_type: str = "NVIDIA_TESLA_T4",
                               gpu_count: int = 1) -> DistributedTrainingConfig:
        """
        Create configuration for multi-node GPU training.
        
        Args:
            display_name: Display name for the training job
            container_uri: Container image URI
            replica_count: Number of worker replicas
            machine_type: Machine type for workers
            gpu_type: GPU type (e.g., NVIDIA_TESLA_T4, NVIDIA_TESLA_V100)
            gpu_count: Number of GPUs per worker
            
        Returns:
            DistributedTrainingConfig object
        """
        worker_pool = WorkerPoolSpec(
            replica_count=replica_count,
            machine_type=machine_type,
            accelerator_type=gpu_type,
            accelerator_count=gpu_count,
            container_spec={
                "image_uri": container_uri
            }
        )
        
        config = DistributedTrainingConfig(
            display_name=display_name,
            distribution_strategy=DistributionStrategy.MULTI_NODE_GPU,
            worker_pool_specs=[worker_pool],
            base_output_dir=f"gs://{self.project_id}-vertex-ai/distributed-training/{display_name}",
            environment_variables={
                "TF_CONFIG": "auto",  # TensorFlow will auto-configure
                "NCCL_DEBUG": "INFO"
            }
        )
        
        logger.info(f"Created multi-GPU config: {display_name} ({replica_count} replicas)")
        return config
    
    def create_tpu_config(self, display_name: str,
                         container_uri: str,
                         tpu_type: str = "v3-8",
                         replica_count: int = 1) -> DistributedTrainingConfig:
        """
        Create configuration for TPU training.
        
        Args:
            display_name: Display name for the training job
            container_uri: Container image URI
            tpu_type: TPU type (e.g., v2-8, v3-8, v4-8)
            replica_count: Number of TPU replicas
            
        Returns:
            DistributedTrainingConfig object
        """
        worker_pool = WorkerPoolSpec(
            replica_count=replica_count,
            machine_type="cloud-tpu",
            accelerator_type=tpu_type,
            accelerator_count=1,
            container_spec={
                "image_uri": container_uri
            }
        )
        
        config = DistributedTrainingConfig(
            display_name=display_name,
            distribution_strategy=DistributionStrategy.MULTI_NODE_TPU,
            worker_pool_specs=[worker_pool],
            base_output_dir=f"gs://{self.project_id}-vertex-ai/distributed-training/{display_name}",
            environment_variables={
                "TPU_CONFIG": "auto",
                "XLA_USE_BF16": "1"
            }
        )
        
        logger.info(f"Created TPU config: {display_name} ({tpu_type})")
        return config
    
    def create_parameter_server_config(self, display_name: str,
                                     container_uri: str,
                                     worker_count: int = 3,
                                     ps_count: int = 2,
                                     chief_count: int = 1) -> DistributedTrainingConfig:
        """
        Create configuration for parameter server training.
        
        Args:
            display_name: Display name for the training job
            container_uri: Container image URI
            worker_count: Number of worker nodes
            ps_count: Number of parameter server nodes
            chief_count: Number of chief nodes
            
        Returns:
            DistributedTrainingConfig object
        """
        # Chief node
        chief_pool = WorkerPoolSpec(
            replica_count=chief_count,
            machine_type="n1-standard-8",
            container_spec={
                "image_uri": container_uri,
                "args": ["--job_type=chief"]
            }
        )
        
        # Worker nodes
        worker_pool = WorkerPoolSpec(
            replica_count=worker_count,
            machine_type="n1-standard-4",
            container_spec={
                "image_uri": container_uri,
                "args": ["--job_type=worker"]
            }
        )
        
        # Parameter server nodes
        ps_pool = WorkerPoolSpec(
            replica_count=ps_count,
            machine_type="n1-standard-2",
            container_spec={
                "image_uri": container_uri,
                "args": ["--job_type=ps"]
            }
        )
        
        config = DistributedTrainingConfig(
            display_name=display_name,
            distribution_strategy=DistributionStrategy.PARAMETER_SERVER,
            worker_pool_specs=[chief_pool, worker_pool, ps_pool],
            base_output_dir=f"gs://{self.project_id}-vertex-ai/distributed-training/{display_name}",
            environment_variables={
                "TF_CONFIG": "auto"
            }
        )
        
        logger.info(f"Created parameter server config: {display_name}")
        return config
    
    def submit_distributed_job(self, config: DistributedTrainingConfig) -> Optional[TrainingJobStatus]:
        """
        Submit a distributed training job.
        
        Args:
            config: DistributedTrainingConfig object
            
        Returns:
            TrainingJobStatus object or None
        """
        if not self.cloud_utils:
            logger.error("Cloud utilities not available")
            return None
        
        try:
            logger.info(f"Submitting distributed training job: {config.display_name}")
            
            # Convert to Vertex AI job specification
            job_spec = self._create_job_spec(config)
            
            # Submit using gcloud CLI (simplified for demo)
            job_id = self._submit_via_gcloud(job_spec, config)
            
            # Create status object
            status = TrainingJobStatus(
                job_id=job_id,
                display_name=config.display_name,
                state="SUBMITTED",
                training_output_dir=config.base_output_dir,
                console_url=f"https://console.cloud.google.com/vertex-ai/locations/{self.location}/training/{job_id}?project={self.project_id}"
            )
            
            logger.info(f"Distributed job submitted: {job_id}")
            return status
            
        except Exception as e:
            logger.error(f"Failed to submit distributed job: {e}")
            return None
    
    def monitor_distributed_job(self, job_id: str) -> TrainingJobStatus:
        """
        Monitor the status of a distributed training job.
        
        Args:
            job_id: Training job ID
            
        Returns:
            TrainingJobStatus object
        """
        try:
            if self.cloud_utils:
                # Get job status using cloud utilities
                status_info = self.cloud_utils.get_job_status(job_id)
                
                status = TrainingJobStatus(
                    job_id=job_id,
                    display_name=status_info.get('displayName', ''),
                    state=status_info.get('state', 'UNKNOWN'),
                    create_time=status_info.get('createTime'),
                    start_time=status_info.get('startTime'),
                    end_time=status_info.get('endTime'),
                    error_message=status_info.get('error', {}).get('message') if status_info.get('error') else None
                )
                
                return status
            
        except Exception as e:
            logger.error(f"Failed to monitor job {job_id}: {e}")
        
        # Return unknown status if monitoring fails
        return TrainingJobStatus(
            job_id=job_id,
            display_name="Unknown",
            state="UNKNOWN"
        )
    
    def list_distributed_jobs(self, limit: int = 50) -> List[TrainingJobStatus]:
        """
        List distributed training jobs.
        
        Args:
            limit: Maximum number of jobs to return
            
        Returns:
            List of TrainingJobStatus objects
        """
        if not self.cloud_utils:
            return []
        
        try:
            jobs = self.cloud_utils.list_training_jobs(limit)
            
            status_list = []
            for job in jobs:
                status = TrainingJobStatus(
                    job_id=job.get('name', '').split('/')[-1],
                    display_name=job.get('displayName', ''),
                    state=job.get('state', 'UNKNOWN'),
                    create_time=job.get('createTime'),
                    start_time=job.get('startTime'),
                    end_time=job.get('endTime')
                )
                status_list.append(status)
            
            return status_list
            
        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            return []
    
    def cancel_distributed_job(self, job_id: str) -> bool:
        """
        Cancel a distributed training job.
        
        Args:
            job_id: Training job ID to cancel
            
        Returns:
            True if successful
        """
        try:
            cmd = [
                'gcloud', 'ai', 'custom-jobs', 'cancel', job_id,
                '--project', self.project_id,
                '--region', self.location,
                '--quiet'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                logger.info(f"Cancelled distributed job: {job_id}")
                return True
            else:
                logger.error(f"Failed to cancel job: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error cancelling job {job_id}: {e}")
            return False
    
    def _create_job_spec(self, config: DistributedTrainingConfig) -> Dict[str, Any]:
        """Create Vertex AI job specification from config."""
        worker_pools = []
        
        for i, pool_spec in enumerate(config.worker_pool_specs):
            worker_pool = {
                "replicaCount": pool_spec.replica_count,
                "machineSpec": {
                    "machineType": pool_spec.machine_type
                }
            }
            
            # Add accelerators if specified
            if pool_spec.accelerator_type and pool_spec.accelerator_count > 0:
                worker_pool["machineSpec"]["acceleratorType"] = pool_spec.accelerator_type
                worker_pool["machineSpec"]["acceleratorCount"] = pool_spec.accelerator_count
            
            # Add disk configuration
            worker_pool["diskSpec"] = {
                "bootDiskType": pool_spec.disk_type,
                "bootDiskSizeGb": pool_spec.disk_size_gb
            }
            
            # Add container spec
            if pool_spec.container_spec:
                worker_pool["containerSpec"] = pool_spec.container_spec.copy()
                worker_pool["containerSpec"]["args"] = config.training_args + worker_pool["containerSpec"].get("args", [])
            
            # Add environment variables
            if config.environment_variables:
                env_vars = [{"name": k, "value": v} for k, v in config.environment_variables.items()]
                worker_pool["containerSpec"]["env"] = env_vars
            
            worker_pools.append(worker_pool)
        
        job_spec = {
            "displayName": config.display_name,
            "jobSpec": {
                "workerPoolSpecs": worker_pools,
                "baseOutputDirectory": {
                    "outputUriPrefix": config.base_output_dir
                },
                "enableWebAccess": config.enable_web_access
            }
        }
        
        # Add service account if specified
        if config.service_account:
            job_spec["jobSpec"]["serviceAccount"] = config.service_account
        
        # Add network configuration if specified
        if config.network:
            job_spec["jobSpec"]["network"] = config.network
        
        # Add scheduling if specified
        if config.scheduling:
            job_spec["jobSpec"]["scheduling"] = config.scheduling
        
        return job_spec
    
    def _submit_via_gcloud(self, job_spec: Dict[str, Any], 
                          config: DistributedTrainingConfig) -> str:
        """Submit job using gcloud CLI."""
        import tempfile
        
        # Create temporary job spec file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(job_spec, f, indent=2)
            spec_file = f.name
        
        try:
            # Generate job ID
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            job_id = f"{config.display_name.lower().replace(' ', '-')}-{timestamp}"
            
            # Submit via gcloud
            cmd = [
                'gcloud', 'ai', 'custom-jobs', 'create',
                '--config', spec_file,
                '--project', self.project_id,
                '--region', self.location,
                '--display-name', config.display_name
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Extract job ID from output (simplified)
                return job_id
            else:
                raise RuntimeError(f"gcloud command failed: {result.stderr}")
                
        finally:
            # Clean up temp file
            os.unlink(spec_file)
    
    def get_distributed_training_recommendations(self, 
                                               dataset_size_gb: float,
                                               model_complexity: str = "medium",
                                               budget_tier: str = "standard") -> Dict[str, Any]:
        """
        Get recommendations for distributed training configuration.
        
        Args:
            dataset_size_gb: Size of training dataset in GB
            model_complexity: Model complexity (simple, medium, complex)
            budget_tier: Budget tier (economy, standard, premium)
            
        Returns:
            Dictionary with recommendations
        """
        recommendations = {
            "dataset_size": dataset_size_gb,
            "model_complexity": model_complexity,
            "budget_tier": budget_tier
        }
        
        # Base recommendations on dataset size
        if dataset_size_gb < 1:
            # Small dataset - single node is sufficient
            recommendations.update({
                "strategy": DistributionStrategy.SINGLE_NODE,
                "machine_type": "n1-standard-4",
                "accelerator": None,
                "estimated_time_hours": 0.5,
                "estimated_cost_usd": 2
            })
        
        elif dataset_size_gb < 10:
            # Medium dataset - consider GPU acceleration
            if budget_tier == "premium":
                recommendations.update({
                    "strategy": DistributionStrategy.SINGLE_NODE,
                    "machine_type": "n1-standard-8",
                    "accelerator": "NVIDIA_TESLA_T4",
                    "accelerator_count": 1,
                    "estimated_time_hours": 1,
                    "estimated_cost_usd": 8
                })
            else:
                recommendations.update({
                    "strategy": DistributionStrategy.SINGLE_NODE,
                    "machine_type": "n1-standard-8",
                    "accelerator": None,
                    "estimated_time_hours": 2,
                    "estimated_cost_usd": 4
                })
        
        elif dataset_size_gb < 100:
            # Large dataset - distributed training recommended
            if budget_tier == "premium":
                recommendations.update({
                    "strategy": DistributionStrategy.MULTI_NODE_GPU,
                    "machine_type": "n1-standard-8",
                    "accelerator": "NVIDIA_TESLA_V100",
                    "accelerator_count": 2,
                    "replica_count": 4,
                    "estimated_time_hours": 3,
                    "estimated_cost_usd": 50
                })
            elif budget_tier == "standard":
                recommendations.update({
                    "strategy": DistributionStrategy.MULTI_NODE_GPU,
                    "machine_type": "n1-standard-8",
                    "accelerator": "NVIDIA_TESLA_T4",
                    "accelerator_count": 1,
                    "replica_count": 2,
                    "estimated_time_hours": 4,
                    "estimated_cost_usd": 25
                })
            else:
                recommendations.update({
                    "strategy": DistributionStrategy.PARAMETER_SERVER,
                    "machine_type": "n1-standard-4",
                    "accelerator": None,
                    "worker_count": 4,
                    "ps_count": 2,
                    "estimated_time_hours": 6,
                    "estimated_cost_usd": 15
                })
        
        else:
            # Very large dataset - TPU or high-end distributed
            if budget_tier == "premium":
                recommendations.update({
                    "strategy": DistributionStrategy.MULTI_NODE_TPU,
                    "tpu_type": "v3-8",
                    "replica_count": 8,
                    "estimated_time_hours": 2,
                    "estimated_cost_usd": 100
                })
            else:
                recommendations.update({
                    "strategy": DistributionStrategy.MULTI_NODE_GPU,
                    "machine_type": "n1-standard-16",
                    "accelerator": "NVIDIA_TESLA_V100",
                    "accelerator_count": 4,
                    "replica_count": 8,
                    "estimated_time_hours": 4,
                    "estimated_cost_usd": 80
                })
        
        # Adjust for model complexity
        if model_complexity == "complex":
            recommendations["estimated_time_hours"] *= 1.5
            recommendations["estimated_cost_usd"] *= 1.3
        elif model_complexity == "simple":
            recommendations["estimated_time_hours"] *= 0.7
            recommendations["estimated_cost_usd"] *= 0.8
        
        return recommendations


def create_distributed_trainer(project_id: str, location: str = "us-central1") -> DistributedTrainer:
    """
    Create a DistributedTrainer instance.
    
    Args:
        project_id: Google Cloud project ID
        location: Vertex AI location
        
    Returns:
        DistributedTrainer instance
    """
    return DistributedTrainer(project_id, location)


def get_machine_type_recommendations() -> Dict[str, Dict[str, Any]]:
    """
    Get recommendations for machine types based on workload.
    
    Returns:
        Dictionary mapping use cases to machine type recommendations
    """
    return {
        "light_training": {
            "machine_type": "n1-standard-4",
            "vcpus": 4,
            "memory_gb": 15,
            "cost_per_hour": 0.19,
            "use_case": "Small datasets, simple models"
        },
        "standard_training": {
            "machine_type": "n1-standard-8",
            "vcpus": 8,
            "memory_gb": 30,
            "cost_per_hour": 0.38,
            "use_case": "Medium datasets, most ML models"
        },
        "intensive_training": {
            "machine_type": "n1-standard-16",
            "vcpus": 16,
            "memory_gb": 60,
            "cost_per_hour": 0.76,
            "use_case": "Large datasets, complex models"
        },
        "gpu_training": {
            "machine_type": "n1-standard-8",
            "vcpus": 8,
            "memory_gb": 30,
            "accelerator": "NVIDIA_TESLA_T4",
            "cost_per_hour": 0.73,
            "use_case": "Deep learning, neural networks"
        },
        "high_end_gpu": {
            "machine_type": "n1-standard-16",
            "vcpus": 16,
            "memory_gb": 60,
            "accelerator": "NVIDIA_TESLA_V100",
            "cost_per_hour": 2.48,
            "use_case": "Large neural networks, research"
        },
        "tpu_training": {
            "machine_type": "cloud-tpu",
            "tpu_type": "v3-8",
            "cost_per_hour": 8.00,
            "use_case": "Very large models, TensorFlow"
        }
    }


def get_accelerator_specifications() -> Dict[str, Dict[str, Any]]:
    """
    Get specifications for available accelerators.
    
    Returns:
        Dictionary mapping accelerator types to specifications
    """
    return {
        "NVIDIA_TESLA_K80": {
            "memory_gb": 24,
            "compute_capability": "3.7",
            "performance_tier": "entry",
            "cost_multiplier": 1.0
        },
        "NVIDIA_TESLA_T4": {
            "memory_gb": 16,
            "compute_capability": "7.5",
            "performance_tier": "standard",
            "cost_multiplier": 1.4
        },
        "NVIDIA_TESLA_V100": {
            "memory_gb": 16,
            "compute_capability": "7.0",
            "performance_tier": "high",
            "cost_multiplier": 2.8
        },
        "NVIDIA_TESLA_P4": {
            "memory_gb": 8,
            "compute_capability": "6.1",
            "performance_tier": "inference",
            "cost_multiplier": 0.9
        },
        "TPU_V2": {
            "memory_gb": 64,
            "performance_tier": "ultra",
            "framework": "TensorFlow",
            "cost_multiplier": 4.5
        },
        "TPU_V3": {
            "memory_gb": 128,
            "performance_tier": "ultra",
            "framework": "TensorFlow",
            "cost_multiplier": 6.0
        }
    }
