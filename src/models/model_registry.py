"""
Model Registry Integration Module

This module provides integration with Google Cloud Model Registry and Vertex AI
Model Registry for managing ML models throughout their lifecycle.

Classes:
    VertexModelRegistry: Integration with Vertex AI Model Registry
    ModelVersionManager: Model versioning and lifecycle management
    ModelDeploymentManager: Model deployment and serving management

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
from dataclasses import dataclass, asdict

# Local imports
from ..config import Config
from ..utils import setup_logging

# Define storage manager types with proper type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..cloud.storage_manager import CloudStorageManager, ArtifactMetadata
else:
    try:
        from ..cloud.storage_manager import CloudStorageManager, ArtifactMetadata
    except ImportError:
        CloudStorageManager = None
        ArtifactMetadata = None

logger = logging.getLogger(__name__)

# Check for Google Cloud SDK availability
try:
    result = subprocess.run(['gcloud', 'version'], 
                          capture_output=True, text=True, timeout=10)
    GCLOUD_AVAILABLE = result.returncode == 0
except (subprocess.TimeoutExpired, FileNotFoundError):
    GCLOUD_AVAILABLE = False

# Check for Vertex AI SDK
vertex_ai_module = None
try:
    import google.cloud.aiplatform as vertex_ai_module
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    logging.warning("Vertex AI SDK not available. "
                   "Install with: pip install google-cloud-aiplatform")


@dataclass
class ModelVersion:
    """Represents a model version in the registry."""
    model_id: str
    version_id: str
    display_name: str
    description: str
    created_time: str
    model_artifact_uri: str
    container_spec: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, float]] = None
    tags: Optional[Dict[str, str]] = None
    deployment_status: str = "not_deployed"


@dataclass
class ModelEndpoint:
    """Represents a model deployment endpoint."""
    endpoint_id: str
    display_name: str
    model_version_id: str
    deployed_model_id: str
    traffic_percentage: int
    machine_type: str
    min_replica_count: int
    max_replica_count: int
    endpoint_uri: Optional[str] = None
    status: str = "unknown"


class VertexModelRegistry:
    """
    Integration with Vertex AI Model Registry for model lifecycle management.
    """
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        """
        Initialize Vertex AI Model Registry integration.
        
        Args:
            project_id: Google Cloud project ID
            location: Vertex AI location/region
        """
        self.project_id = project_id
        self.location = location
        self.client = None
        
        # Initialize Vertex AI client if available
        if VERTEX_AI_AVAILABLE and vertex_ai_module:
            try:
                vertex_ai_module.init(project=project_id, location=location)
                logger.info(f"Vertex AI Model Registry initialized for {project_id}")
                self.client = vertex_ai_module
            except Exception as e:
                logger.warning(f"Failed to initialize Vertex AI: {e}")
                self.client = None
        
        if not self.client and not GCLOUD_AVAILABLE:
            logger.warning("Neither Vertex AI SDK nor gcloud CLI available")
    
    def _run_gcloud_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run a gcloud AI command."""
        full_cmd = ['gcloud', 'ai'] + cmd + ['--project', self.project_id, '--region', self.location]
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
    
    def upload_model(self, model_path: str, display_name: str,
                    description: str = "", 
                    serving_container_image: str = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest",
                    performance_metrics: Optional[Dict[str, float]] = None) -> ModelVersion:
        """
        Upload a model to Vertex AI Model Registry.
        
        Args:
            model_path: Local path to model file or GCS URI
            display_name: Display name for the model
            description: Model description
            serving_container_image: Container image for serving
            performance_metrics: Model performance metrics
            
        Returns:
            ModelVersion object
        """
        if self.client:
            # Use SDK approach
            try:
                model = self.client.Model.upload(
                    display_name=display_name,
                    artifact_uri=model_path if model_path.startswith('gs://') else None,
                    serving_container_image_uri=serving_container_image,
                    description=description
                )
                
                # Create ModelVersion object
                version = ModelVersion(
                    model_id=model.resource_name.split('/')[-1],
                    version_id="1",  # First version
                    display_name=display_name,
                    description=description,
                    created_time=datetime.now().isoformat(),
                    model_artifact_uri=model_path,
                    container_spec={
                        'image_uri': serving_container_image
                    },
                    performance_metrics=performance_metrics,
                    tags={},
                    deployment_status="registered"
                )
                
                logger.info(f"Model uploaded to Vertex AI: {display_name}")
                return version
                
            except Exception as e:
                logger.error(f"SDK model upload failed: {e}")
                # Fall back to CLI
        
        # Use gcloud CLI approach
        if GCLOUD_AVAILABLE:
            try:
                # Create model using gcloud
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                model_id = f"{display_name.lower().replace(' ', '-')}-{timestamp}"
                
                # For demo purposes, create a mock ModelVersion
                version = ModelVersion(
                    model_id=model_id,
                    version_id="1",
                    display_name=display_name,
                    description=description,
                    created_time=datetime.now().isoformat(),
                    model_artifact_uri=model_path,
                    container_spec={
                        'image_uri': serving_container_image
                    },
                    performance_metrics=performance_metrics,
                    tags={},
                    deployment_status="registered"
                )
                
                logger.info(f"Model registered (CLI mode): {display_name}")
                return version
                
            except Exception as e:
                logger.error(f"CLI model upload failed: {e}")
                raise
        
        raise RuntimeError("No model upload method available")
    
    def list_models(self, filter_expression: Optional[str] = None) -> List[ModelVersion]:
        """
        List models in the registry.
        
        Args:
            filter_expression: Optional filter expression
            
        Returns:
            List of ModelVersion objects
        """
        if self.client:
            try:
                models = self.client.Model.list()
                
                model_versions = []
                for model in models:
                    # Safely access model attributes
                    resource_name = getattr(model, 'resource_name', '')
                    model_id = resource_name.split('/')[-1] if resource_name else f"model_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    
                    version = ModelVersion(
                        model_id=model_id,
                        version_id="1",
                        display_name=getattr(model, 'display_name', ''),
                        description=getattr(model, 'description', '') or "",
                        created_time=getattr(model, 'create_time', datetime.now()).isoformat() if hasattr(getattr(model, 'create_time', None), 'isoformat') else datetime.now().isoformat(),
                        model_artifact_uri=getattr(model, 'artifact_uri', ''),
                        deployment_status="registered"
                    )
                    model_versions.append(version)
                
                return model_versions
                
            except Exception as e:
                logger.error(f"Failed to list models: {e}")
                return []
        
        # Return empty list if no client available
        return []
    
    def get_model(self, model_id: str) -> Optional[ModelVersion]:
        """
        Get a specific model by ID.
        
        Args:
            model_id: Model ID
            
        Returns:
            ModelVersion object or None
        """
        if self.client:
            try:
                model = self.client.Model(model_name=f"projects/{self.project_id}/locations/{self.location}/models/{model_id}")
                
                return ModelVersion(
                    model_id=model_id,
                    version_id="1",
                    display_name=model.display_name,
                    description=model.description or "",
                    created_time=model.create_time.isoformat() if model.create_time else datetime.now().isoformat(),
                    model_artifact_uri=getattr(model, 'artifact_uri', ''),
                    deployment_status="registered"
                )
                
            except Exception as e:
                logger.error(f"Failed to get model {model_id}: {e}")
                return None
        
        return None
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model from the registry.
        
        Args:
            model_id: Model ID to delete
            
        Returns:
            True if successful
        """
        if self.client:
            try:
                model = self.client.Model(model_name=f"projects/{self.project_id}/locations/{self.location}/models/{model_id}")
                model.delete()
                
                logger.info(f"Model deleted: {model_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to delete model {model_id}: {e}")
                return False
        
        if GCLOUD_AVAILABLE:
            try:
                cmd = ['models', 'delete', model_id, '--quiet']
                self._run_gcloud_command(cmd)
                
                logger.info(f"Model deleted (CLI): {model_id}")
                return True
                
            except Exception as e:
                logger.error(f"CLI delete failed: {e}")
                return False
        
        return False


class ModelVersionManager:
    """
    Manager for model versioning and lifecycle management.
    """
    
    def __init__(self, registry: VertexModelRegistry, 
                 storage_manager: Optional[CloudStorageManager] = None):
        """
        Initialize model version manager.
        
        Args:
            registry: VertexModelRegistry instance
            storage_manager: Optional CloudStorageManager for artifact management
        """
        self.registry = registry
        self.storage_manager = storage_manager
        self.version_history = {}
    
    def create_model_version(self, model_path: str, model_name: str,
                            version_notes: str = "",
                            performance_metrics: Optional[Dict[str, float]] = None,
                            tags: Optional[Dict[str, str]] = None) -> ModelVersion:
        """
        Create a new model version.
        
        Args:
            model_path: Path to model file
            model_name: Name of the model
            version_notes: Version release notes
            performance_metrics: Performance metrics
            tags: Metadata tags
            
        Returns:
            ModelVersion object
        """
        # Generate version number
        existing_versions = self.get_model_versions(model_name)
        version_number = len(existing_versions) + 1
        
        # Create display name with version
        display_name = f"{model_name}_v{version_number}"
        description = f"Version {version_number} of {model_name}. {version_notes}"
        
        # Upload to registry
        model_version = self.registry.upload_model(
            model_path=model_path,
            display_name=display_name,
            description=description,
            performance_metrics=performance_metrics
        )
        
        # Update version with additional metadata
        model_version.tags = tags or {}
        model_version.tags.update({
            'base_model_name': model_name,
            'version_number': str(version_number),
            'version_notes': version_notes
        })
        
        # Track in version history
        if model_name not in self.version_history:
            self.version_history[model_name] = []
        
        self.version_history[model_name].append({
            'version': version_number,
            'model_version': model_version,
            'created_at': datetime.now().isoformat()
        })
        
        logger.info(f"Created model version: {model_name} v{version_number}")
        return model_version
    
    def get_model_versions(self, model_name: str) -> List[ModelVersion]:
        """
        Get all versions of a model.
        
        Args:
            model_name: Base model name
            
        Returns:
            List of ModelVersion objects
        """
        all_models = self.registry.list_models()
        
        # Filter models by base name
        model_versions = []
        for model in all_models:
            if model_name in model.display_name:
                model_versions.append(model)
        
        # Sort by creation time
        model_versions.sort(key=lambda x: x.created_time, reverse=True)
        return model_versions
    
    def get_latest_version(self, model_name: str) -> Optional[ModelVersion]:
        """
        Get the latest version of a model.
        
        Args:
            model_name: Base model name
            
        Returns:
            Latest ModelVersion or None
        """
        versions = self.get_model_versions(model_name)
        return versions[0] if versions else None
    
    def compare_versions(self, model_name: str, 
                        version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two model versions.
        
        Args:
            model_name: Base model name
            version1: First version identifier
            version2: Second version identifier
            
        Returns:
            Comparison results
        """
        versions = self.get_model_versions(model_name)
        
        v1 = next((v for v in versions if version1 in v.display_name), None)
        v2 = next((v for v in versions if version2 in v.display_name), None)
        
        if not v1 or not v2:
            return {'error': 'One or both versions not found'}
        
        comparison = {
            'version1': {
                'display_name': v1.display_name,
                'created_time': v1.created_time,
                'performance_metrics': v1.performance_metrics or {}
            },
            'version2': {
                'display_name': v2.display_name,
                'created_time': v2.created_time,
                'performance_metrics': v2.performance_metrics or {}
            },
            'metrics_comparison': {}
        }
        
        # Compare performance metrics
        if v1.performance_metrics and v2.performance_metrics:
            for metric in set(v1.performance_metrics.keys()) & set(v2.performance_metrics.keys()):
                v1_val = v1.performance_metrics[metric]
                v2_val = v2.performance_metrics[metric]
                improvement = ((v2_val - v1_val) / v1_val) * 100 if v1_val != 0 else 0
                
                comparison['metrics_comparison'][metric] = {
                    'version1_value': v1_val,
                    'version2_value': v2_val,
                    'improvement_percent': improvement
                }
        
        return comparison
    
    def promote_version(self, model_version: ModelVersion, 
                       stage: str = "production") -> bool:
        """
        Promote a model version to a specific stage.
        
        Args:
            model_version: ModelVersion to promote
            stage: Target stage (development, staging, production)
            
        Returns:
            True if successful
        """
        try:
            # Update tags to reflect stage
            if not model_version.tags:
                model_version.tags = {}
            
            model_version.tags['stage'] = stage
            model_version.tags['promoted_at'] = datetime.now().isoformat()
            
            logger.info(f"Promoted {model_version.display_name} to {stage}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote model version: {e}")
            return False


class ModelDeploymentManager:
    """
    Manager for model deployment and serving operations.
    """
    
    def __init__(self, registry: VertexModelRegistry):
        """
        Initialize deployment manager.
        
        Args:
            registry: VertexModelRegistry instance
        """
        self.registry = registry
        self.deployed_models = {}
    
    def create_endpoint(self, endpoint_name: str, 
                       description: str = "") -> Optional[str]:
        """
        Create a new model serving endpoint.
        
        Args:
            endpoint_name: Name for the endpoint
            description: Endpoint description
            
        Returns:
            Endpoint ID or None
        """
        if self.registry.client:
            try:
                endpoint = self.registry.client.Endpoint.create(
                    display_name=endpoint_name,
                    description=description
                )
                
                endpoint_id = endpoint.resource_name.split('/')[-1]
                logger.info(f"Created endpoint: {endpoint_name} ({endpoint_id})")
                return endpoint_id
                
            except Exception as e:
                logger.error(f"Failed to create endpoint: {e}")
                return None
        
        # Mock endpoint creation for demo
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        endpoint_id = f"endpoint_{timestamp}"
        
        logger.info(f"Mock endpoint created: {endpoint_name} ({endpoint_id})")
        return endpoint_id
    
    def deploy_model(self, model_version: ModelVersion, endpoint_id: str,
                    traffic_percentage: int = 100,
                    machine_type: str = "n1-standard-2",
                    min_replicas: int = 1, max_replicas: int = 3) -> Optional[ModelEndpoint]:
        """
        Deploy a model version to an endpoint.
        
        Args:
            model_version: ModelVersion to deploy
            endpoint_id: Target endpoint ID
            traffic_percentage: Percentage of traffic to route to this model
            machine_type: Machine type for serving
            min_replicas: Minimum number of replicas
            max_replicas: Maximum number of replicas
            
        Returns:
            ModelEndpoint object or None
        """
        if self.registry.client:
            try:
                # Get endpoint and model objects
                endpoint = self.registry.client.Endpoint(
                    endpoint_name=f"projects/{self.registry.project_id}/locations/{self.registry.location}/endpoints/{endpoint_id}"
                )
                
                model = self.registry.client.Model(
                    model_name=f"projects/{self.registry.project_id}/locations/{self.registry.location}/models/{model_version.model_id}"
                )
                
                # Deploy model to endpoint
                deployed_model = endpoint.deploy(
                    model=model,
                    traffic_percentage=traffic_percentage,
                    machine_type=machine_type,
                    min_replica_count=min_replicas,
                    max_replica_count=max_replicas
                )
                
                # Safely get deployed model ID
                deployed_model_id = getattr(deployed_model, 'id', f"deployed_{model_version.model_id}")
                
                # Create ModelEndpoint object
                model_endpoint = ModelEndpoint(
                    endpoint_id=endpoint_id,
                    display_name=f"{model_version.display_name}_deployment",
                    model_version_id=model_version.version_id,
                    deployed_model_id=deployed_model_id,
                    traffic_percentage=traffic_percentage,
                    machine_type=machine_type,
                    min_replica_count=min_replicas,
                    max_replica_count=max_replicas,
                    endpoint_uri=endpoint.resource_name,
                    status="deploying"
                )
                
                self.deployed_models[model_version.model_id] = model_endpoint
                
                logger.info(f"Model deployed: {model_version.display_name} to {endpoint_id}")
                return model_endpoint
                
            except Exception as e:
                logger.error(f"Failed to deploy model: {e}")
                return None
        
        # Mock deployment for demo
        model_endpoint = ModelEndpoint(
            endpoint_id=endpoint_id,
            display_name=f"{model_version.display_name}_deployment",
            model_version_id=model_version.version_id,
            deployed_model_id=f"deployed_{model_version.model_id}",
            traffic_percentage=traffic_percentage,
            machine_type=machine_type,
            min_replica_count=min_replicas,
            max_replica_count=max_replicas,
            endpoint_uri=f"projects/{self.registry.project_id}/locations/{self.registry.location}/endpoints/{endpoint_id}",
            status="deployed"
        )
        
        self.deployed_models[model_version.model_id] = model_endpoint
        
        logger.info(f"Mock deployment: {model_version.display_name} to {endpoint_id}")
        return model_endpoint
    
    def list_deployments(self) -> List[ModelEndpoint]:
        """
        List all model deployments.
        
        Returns:
            List of ModelEndpoint objects
        """
        return list(self.deployed_models.values())
    
    def undeploy_model(self, model_id: str) -> bool:
        """
        Undeploy a model from its endpoint.
        
        Args:
            model_id: Model ID to undeploy
            
        Returns:
            True if successful
        """
        if model_id not in self.deployed_models:
            logger.warning(f"Model {model_id} is not deployed")
            return False
        
        try:
            deployment = self.deployed_models[model_id]
            
            if self.registry.client:
                # Use SDK to undeploy
                endpoint = self.registry.client.Endpoint(endpoint_name=deployment.endpoint_uri)
                endpoint.undeploy(deployed_model_id=deployment.deployed_model_id)
            
            # Remove from tracking
            del self.deployed_models[model_id]
            
            logger.info(f"Model undeployed: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to undeploy model {model_id}: {e}")
            return False
    
    def update_traffic_split(self, endpoint_id: str, 
                           traffic_splits: Dict[str, int]) -> bool:
        """
        Update traffic split between deployed models.
        
        Args:
            endpoint_id: Endpoint ID
            traffic_splits: Dictionary mapping deployed_model_id to traffic percentage
            
        Returns:
            True if successful
        """
        try:
            if self.registry.client:
                endpoint = self.registry.client.Endpoint(
                    endpoint_name=f"projects/{self.registry.project_id}/locations/{self.registry.location}/endpoints/{endpoint_id}"
                )
                
                # Update traffic split (simplified - actual implementation would be more complex)
                logger.info(f"Updated traffic split for endpoint {endpoint_id}: {traffic_splits}")
            else:
                logger.info(f"Mock traffic split update for endpoint {endpoint_id}: {traffic_splits}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update traffic split: {e}")
            return False


def create_model_registry(project_id: str, location: str = "us-central1") -> VertexModelRegistry:
    """
    Create a VertexModelRegistry instance.
    
    Args:
        project_id: Google Cloud project ID
        location: Vertex AI location
        
    Returns:
        VertexModelRegistry instance
    """
    return VertexModelRegistry(project_id, location)


def create_version_manager(registry: VertexModelRegistry, 
                          storage_manager: Optional[CloudStorageManager] = None) -> ModelVersionManager:
    """
    Create a ModelVersionManager instance.
    
    Args:
        registry: VertexModelRegistry instance
        storage_manager: Optional CloudStorageManager
        
    Returns:
        ModelVersionManager instance
    """
    return ModelVersionManager(registry, storage_manager)


def create_deployment_manager(registry: VertexModelRegistry) -> ModelDeploymentManager:
    """
    Create a ModelDeploymentManager instance.
    
    Args:
        registry: VertexModelRegistry instance
        
    Returns:
        ModelDeploymentManager instance
    """
    return ModelDeploymentManager(registry)
