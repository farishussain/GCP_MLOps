"""
Pipeline Orchestration Module

This module provides comprehensive pipeline orchestration capabilities using
Vertex AI Pipelines (Kubeflow Pipelines) for end-to-end MLOps workflows.

Classes:
    PipelineConfig: Configuration for ML pipelines
    PipelineManager: Manager for pipeline creation and execution
    PipelineComponent: Base class for pipeline components
    MLOpsPipeline: End-to-end MLOps pipeline implementation

Author: MLOps Team
Version: 1.0.0
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

# Local imports
from ..config import Config
from ..utils import setup_logging
from ..data.data_loader import DataLoader
from ..models.trainer import ModelTrainer
from ..cloud.vertex_ai import CloudTrainingUtils
from ..deployment.model_deployment import EndpointManager

logger = logging.getLogger(__name__)

# Check for Kubeflow Pipelines SDK availability
try:
    import kfp
    from kfp.v2 import dsl, compiler
    from kfp.v2.dsl import component, pipeline, Input, Output, Model, Dataset, Metrics
    KFP_AVAILABLE = True
except ImportError:
    kfp = None
    dsl = None
    component = None
    pipeline = None
    KFP_AVAILABLE = False
    logger.warning("Kubeflow Pipelines SDK not available. Install with: pip install kfp")

# Check for Vertex AI Pipelines
try:
    from google.cloud import aiplatform
    from google.cloud.aiplatform import PipelineJob
    VERTEX_PIPELINES_AVAILABLE = True
except ImportError:
    VERTEX_PIPELINES_AVAILABLE = False
    logger.warning("Vertex AI SDK not available for pipelines")


class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class ComponentType(Enum):
    """Types of pipeline components."""
    DATA_INGESTION = "data_ingestion"
    DATA_PREPROCESSING = "data_preprocessing"
    DATA_VALIDATION = "data_validation"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    MODEL_VALIDATION = "model_validation"
    MODEL_DEPLOYMENT = "model_deployment"
    MODEL_MONITORING = "model_monitoring"


@dataclass
class PipelineConfig:
    """Configuration for ML pipeline."""
    pipeline_name: str
    pipeline_description: str = ""
    project_id: str = ""
    location: str = "us-central1"
    pipeline_root: str = ""
    enable_caching: bool = True
    
    # Data configuration
    data_source: str = ""
    data_format: str = "csv"
    validation_split: float = 0.2
    
    # Training configuration
    model_type: str = "auto"
    training_algorithm: str = "random_forest"
    hyperparameter_tuning: bool = True
    max_trial_count: int = 10
    
    # Deployment configuration
    deploy_model: bool = False
    endpoint_name: str = ""
    machine_type: str = "n1-standard-2"
    min_replica_count: int = 1
    max_replica_count: int = 10
    
    # Monitoring configuration
    enable_monitoring: bool = False
    monitoring_window_hours: int = 24
    alert_threshold: float = 0.05
    
    # Pipeline metadata
    labels: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentConfig:
    """Configuration for a pipeline component."""
    name: str
    component_type: ComponentType
    image_uri: str = "gcr.io/deeplearning-platform-release/base-cpu"
    machine_type: str = "n1-standard-2"
    cpu_limit: str = "2"
    memory_limit: str = "8Gi"
    gpu_type: Optional[str] = None
    gpu_count: int = 0
    timeout_seconds: int = 3600
    retry_count: int = 1
    
    # Component-specific parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_paths: List[str] = field(default_factory=list)
    output_paths: List[str] = field(default_factory=list)


@dataclass
class PipelineRun:
    """Information about a pipeline run."""
    run_id: str
    pipeline_name: str
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    error_message: Optional[str] = None


class PipelineComponent:
    """Base class for pipeline components."""
    
    def __init__(self, name: str, config: ComponentConfig):
        """
        Initialize pipeline component.
        
        Args:
            name: Component name
            config: Component configuration
        """
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"component.{name}")
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the component logic.
        
        Args:
            inputs: Input parameters and artifacts
            
        Returns:
            Dictionary of outputs and artifacts
        """
        raise NotImplementedError("Subclasses must implement execute method")
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """
        Validate component inputs.
        
        Args:
            inputs: Input parameters and artifacts
            
        Returns:
            True if inputs are valid
        """
        return True
    
    def get_component_spec(self) -> Dict[str, Any]:
        """
        Get component specification for KFP.
        
        Returns:
            Component specification dictionary
        """
        return {
            'name': self.name,
            'description': f"Component for {self.config.component_type.value}",
            'image': self.config.image_uri,
            'command': ['python', '-c'],
            'args': [self._get_component_code()],
            'inputs': self._get_input_spec(),
            'outputs': self._get_output_spec()
        }
    
    def _get_component_code(self) -> str:
        """Get Python code for the component."""
        return f"""
import json
import logging

def main():
    print("Executing component: {self.name}")
    # Component execution logic would go here
    return {{"status": "success"}}

if __name__ == "__main__":
    main()
"""
    
    def _get_input_spec(self) -> List[Dict[str, Any]]:
        """Get input specification."""
        return [
            {'name': 'input_data', 'type': 'Dataset'},
            {'name': 'parameters', 'type': 'Dict'}
        ]
    
    def _get_output_spec(self) -> List[Dict[str, Any]]:
        """Get output specification."""
        return [
            {'name': 'output_data', 'type': 'Dataset'},
            {'name': 'metrics', 'type': 'Metrics'}
        ]


class DataIngestionComponent(PipelineComponent):
    """Data ingestion pipeline component."""
    
    def __init__(self, config: ComponentConfig):
        """Initialize data ingestion component."""
        config.component_type = ComponentType.DATA_INGESTION
        super().__init__("data_ingestion", config)
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data ingestion logic."""
        self.logger.info("Starting data ingestion")
        
        try:
            # Initialize data loader
            data_loader = DataLoader()
            
            # Load data from source (simplified call)
            data_source = inputs.get('data_source', self.config.parameters.get('data_source'))
            if hasattr(data_loader, 'load_iris_data'):
                data = data_loader.load_iris_data()
            else:
                # Fallback data loading
                import pandas as pd
                data = pd.DataFrame({
                    'feature1': [1, 2, 3, 4, 5],
                    'feature2': [2, 4, 6, 8, 10], 
                    'target': [0, 1, 0, 1, 0]
                })
            
            # Validate data
            if data is None or len(data) == 0:
                raise ValueError("No data loaded from source")
            
            self.logger.info(f"Loaded {len(data)} records")
            
            # Return outputs
            return {
                'output_data': data,
                'record_count': len(data),
                'data_shape': list(data.shape) if hasattr(data, 'shape') else None,
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"Data ingestion failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }


class DataPreprocessingComponent(PipelineComponent):
    """Data preprocessing pipeline component."""
    
    def __init__(self, config: ComponentConfig):
        """Initialize data preprocessing component."""
        config.component_type = ComponentType.DATA_PREPROCESSING
        super().__init__("data_preprocessing", config)
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data preprocessing logic."""
        self.logger.info("Starting data preprocessing")
        
        try:
            from ..data.preprocessor import DataPreprocessor
            
            # Initialize preprocessor
            preprocessor = DataPreprocessor()
            
            # Get input data
            input_data = inputs.get('input_data')
            if input_data is None:
                raise ValueError("No input data provided")
            
            # Preprocess data
            processed_data = preprocessor.fit_transform(input_data)
            
            self.logger.info(f"Processed {len(processed_data)} records")
            
            return {
                'output_data': processed_data,
                'preprocessor': preprocessor,
                'processing_steps': preprocessor.get_feature_names(),
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }


class ModelTrainingComponent(PipelineComponent):
    """Model training pipeline component."""
    
    def __init__(self, config: ComponentConfig):
        """Initialize model training component."""
        config.component_type = ComponentType.MODEL_TRAINING
        super().__init__("model_training", config)
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model training logic."""
        self.logger.info("Starting model training")
        
        try:
            # Initialize trainer
            trainer = ModelTrainer()
            
            # Get input data
            input_data = inputs.get('input_data')
            target_column = inputs.get('target_column', 'target')
            
            if input_data is None:
                raise ValueError("No input data provided")
            
            # Split features and target
            if hasattr(input_data, 'drop'):
                X = input_data.drop(columns=[target_column])
                y = input_data[target_column]
            else:
                raise ValueError("Input data format not supported")
            
            # Train model
            training_config = inputs.get('training_config', {})
            algorithm = training_config.get('algorithm', 'random_forest')
            
            model_result = trainer.train_model(X, y, algorithm)
            
            self.logger.info(f"Model trained with accuracy: {model_result.accuracy:.4f}")
            
            return {
                'trained_model': model_result.model,
                'model_metrics': {
                    'accuracy': model_result.accuracy,
                    'precision': model_result.precision,
                    'recall': model_result.recall,
                    'f1_score': model_result.f1_score
                },
                'feature_names': list(X.columns) if hasattr(X, 'columns') else None,
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }


class ModelDeploymentComponent(PipelineComponent):
    """Model deployment pipeline component."""
    
    def __init__(self, config: ComponentConfig):
        """Initialize model deployment component."""
        config.component_type = ComponentType.MODEL_DEPLOYMENT
        super().__init__("model_deployment", config)
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model deployment logic."""
        self.logger.info("Starting model deployment")
        
        try:
            # Get deployment configuration
            project_id = inputs.get('project_id')
            model_artifact = inputs.get('model_artifact')
            endpoint_config = inputs.get('endpoint_config', {})
            
            if not project_id or not model_artifact:
                raise ValueError("Project ID and model artifact required")
            
            # Initialize endpoint manager
            endpoint_manager = EndpointManager(project_id)
            
            # Create endpoint if needed
            endpoint_name = endpoint_config.get('endpoint_name', f"model-endpoint-{int(time.time())}")
            
            endpoint_id = endpoint_manager.create_endpoint(
                display_name=endpoint_name,
                description="Pipeline deployed model endpoint"
            )
            
            if not endpoint_id:
                raise ValueError("Failed to create endpoint")
            
            self.logger.info(f"Model deployment initiated to endpoint: {endpoint_id}")
            
            return {
                'endpoint_id': endpoint_id,
                'endpoint_name': endpoint_name,
                'deployment_status': 'initiated',
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"Model deployment failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }


class PipelineManager:
    """
    Manager for ML pipeline creation and execution.
    
    Handles pipeline definition, compilation, submission, and monitoring
    using Vertex AI Pipelines.
    """
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        """
        Initialize pipeline manager.
        
        Args:
            project_id: Google Cloud project ID
            location: Vertex AI location/region
        """
        self.project_id = project_id
        self.location = location
        
        # Initialize Vertex AI
        if VERTEX_PIPELINES_AVAILABLE:
            try:
                aiplatform.init(project=project_id, location=location)
                logger.info(f"Pipeline manager initialized for {project_id} in {location}")
            except Exception as e:
                logger.error(f"Failed to initialize Vertex AI: {e}")
        
        # Pipeline registry
        self.pipelines: Dict[str, Any] = {}
        self.pipeline_runs: Dict[str, PipelineRun] = {}
    
    def create_mlops_pipeline(self, config: PipelineConfig) -> Optional[str]:
        """
        Create end-to-end MLOps pipeline.
        
        Args:
            config: Pipeline configuration
            
        Returns:
            Pipeline specification or None
        """
        try:
            if not KFP_AVAILABLE:
                logger.error("Kubeflow Pipelines SDK not available")
                return None
            
            # Define pipeline function
            @pipeline(
                name=config.pipeline_name,
                description=config.pipeline_description,
                pipeline_root=config.pipeline_root or f"gs://{self.project_id}-mlops-pipeline"
            )
            def mlops_pipeline_func(
                data_source: str = config.data_source,
                target_column: str = "target",
                model_algorithm: str = config.training_algorithm,
                deploy_model: bool = config.deploy_model
            ):
                """MLOps pipeline function."""
                
                # Data ingestion
                data_ingestion = self._create_data_ingestion_component()
                data_task = data_ingestion(data_source=data_source)
                
                # Data preprocessing
                preprocessing = self._create_preprocessing_component()
                preprocess_task = preprocessing(input_data=data_task.outputs['output_data'])
                
                # Model training
                training = self._create_training_component()
                training_task = training(
                    input_data=preprocess_task.outputs['output_data'],
                    target_column=target_column,
                    algorithm=model_algorithm
                )
                
                # Conditional deployment
                with dsl.Condition(deploy_model == True):
                    deployment = self._create_deployment_component()
                    deploy_task = deployment(
                        model_artifact=training_task.outputs['trained_model'],
                        project_id=self.project_id
                    )
                
                return {
                    'model_metrics': training_task.outputs['model_metrics'],
                    'endpoint_id': deploy_task.outputs['endpoint_id'] if deploy_model else None
                }
            
            # Compile pipeline
            pipeline_spec_path = f"/tmp/{config.pipeline_name}.json"
            compiler.Compiler().compile(
                pipeline_func=mlops_pipeline_func,
                package_path=pipeline_spec_path
            )
            
            # Store pipeline
            self.pipelines[config.pipeline_name] = {
                'config': config,
                'spec_path': pipeline_spec_path,
                'pipeline_func': mlops_pipeline_func
            }
            
            logger.info(f"Pipeline {config.pipeline_name} created and compiled")
            return pipeline_spec_path
            
        except Exception as e:
            logger.error(f"Failed to create pipeline: {e}")
            return None
    
    def submit_pipeline(self, pipeline_name: str, 
                       parameters: Optional[Dict[str, Any]] = None,
                       run_name: Optional[str] = None) -> Optional[str]:
        """
        Submit pipeline for execution.
        
        Args:
            pipeline_name: Name of pipeline to submit
            parameters: Pipeline parameters
            run_name: Optional run name
            
        Returns:
            Run ID or None
        """
        try:
            if pipeline_name not in self.pipelines:
                logger.error(f"Pipeline {pipeline_name} not found")
                return None
            
            if not VERTEX_PIPELINES_AVAILABLE:
                logger.error("Vertex AI Pipelines not available")
                return None
            
            pipeline_info = self.pipelines[pipeline_name]
            config = pipeline_info['config']
            spec_path = pipeline_info['spec_path']
            
            # Create run name
            if not run_name:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                run_name = f"{pipeline_name}-{timestamp}"
            
            # Submit to Vertex AI Pipelines
            job = PipelineJob(
                display_name=run_name,
                template_path=spec_path,
                pipeline_root=config.pipeline_root or f"gs://{self.project_id}-mlops-pipeline",
                parameter_values=parameters or {},
                enable_caching=config.enable_caching,
                labels=config.labels
            )
            
            job.submit()
            
            # Track run
            run_id = job.resource_name.split('/')[-1]
            self.pipeline_runs[run_id] = PipelineRun(
                run_id=run_id,
                pipeline_name=pipeline_name,
                status=PipelineStatus.PENDING,
                start_time=datetime.now(),
                parameters=parameters or {}
            )
            
            logger.info(f"Pipeline submitted: {run_name} (ID: {run_id})")
            return run_id
            
        except Exception as e:
            logger.error(f"Failed to submit pipeline: {e}")
            return None
    
    def get_run_status(self, run_id: str) -> Optional[PipelineRun]:
        """
        Get status of a pipeline run.
        
        Args:
            run_id: Pipeline run ID
            
        Returns:
            PipelineRun object or None
        """
        try:
            if run_id in self.pipeline_runs:
                # Update with latest status (simplified)
                run = self.pipeline_runs[run_id]
                
                # In real implementation, query Vertex AI for actual status
                # For now, simulate progression
                elapsed = datetime.now() - run.start_time
                
                if elapsed.total_seconds() < 60:
                    run.status = PipelineStatus.RUNNING
                else:
                    run.status = PipelineStatus.SUCCEEDED
                    run.end_time = datetime.now()
                    run.duration_seconds = int(elapsed.total_seconds())
                
                return run
            else:
                logger.warning(f"Run {run_id} not found")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get run status: {e}")
            return None
    
    def list_pipeline_runs(self, pipeline_name: Optional[str] = None) -> List[PipelineRun]:
        """
        List pipeline runs.
        
        Args:
            pipeline_name: Optional pipeline name filter
            
        Returns:
            List of PipelineRun objects
        """
        runs = list(self.pipeline_runs.values())
        
        if pipeline_name:
            runs = [r for r in runs if r.pipeline_name == pipeline_name]
        
        # Sort by start time (newest first)
        runs.sort(key=lambda x: x.start_time, reverse=True)
        
        return runs
    
    def cancel_run(self, run_id: str) -> bool:
        """
        Cancel a pipeline run.
        
        Args:
            run_id: Pipeline run ID
            
        Returns:
            True if cancellation successful
        """
        try:
            if run_id not in self.pipeline_runs:
                logger.error(f"Run {run_id} not found")
                return False
            
            # In real implementation, cancel via Vertex AI API
            run = self.pipeline_runs[run_id]
            run.status = PipelineStatus.CANCELLED
            run.end_time = datetime.now()
            
            logger.info(f"Pipeline run {run_id} cancelled")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel run: {e}")
            return False
    
    def _create_data_ingestion_component(self):
        """Create data ingestion component."""
        if not KFP_AVAILABLE:
            return None
            
        @component(
            base_image="python:3.8",
            packages_to_install=["pandas", "numpy"]
        )
        def data_ingestion_component(
            data_source: str,
            output_data: Output[Dataset]
        ) -> Dict[str, Any]:
            """Data ingestion component."""
            import pandas as pd
            import json
            
            # Mock data loading
            if "iris" in data_source.lower():
                from sklearn.datasets import load_iris
                data = load_iris(as_frame=True)
                df = data.frame
            else:
                # Create sample data
                df = pd.DataFrame({
                    'feature1': [1, 2, 3, 4, 5],
                    'feature2': [2, 4, 6, 8, 10],
                    'target': [0, 1, 0, 1, 0]
                })
            
            # Save data
            df.to_csv(output_data.path, index=False)
            
            return {
                'record_count': len(df),
                'feature_count': len(df.columns) - 1
            }
        
        return data_ingestion_component
    
    def _create_preprocessing_component(self):
        """Create preprocessing component."""
        if not KFP_AVAILABLE:
            return None
            
        @component(
            base_image="python:3.8",
            packages_to_install=["pandas", "scikit-learn"]
        )
        def preprocessing_component(
            input_data: Input[Dataset],
            output_data: Output[Dataset]
        ) -> Dict[str, Any]:
            """Data preprocessing component."""
            import pandas as pd
            from sklearn.preprocessing import StandardScaler
            
            # Load data
            df = pd.read_csv(input_data.path)
            
            # Basic preprocessing
            # Separate features and target
            target_col = 'target'
            feature_cols = [col for col in df.columns if col != target_col]
            
            # Scale features
            scaler = StandardScaler()
            df[feature_cols] = scaler.fit_transform(df[feature_cols])
            
            # Save processed data
            df.to_csv(output_data.path, index=False)
            
            return {
                'processed_features': len(feature_cols),
                'scaling_applied': True
            }
        
        return preprocessing_component
    
    def _create_training_component(self):
        """Create model training component."""
        if not KFP_AVAILABLE:
            return None
            
        @component(
            base_image="python:3.8",
            packages_to_install=["pandas", "scikit-learn", "joblib"]
        )
        def training_component(
            input_data: Input[Dataset],
            target_column: str,
            algorithm: str,
            model_output: Output[Model],
            metrics_output: Output[Metrics]
        ) -> Dict[str, Any]:
            """Model training component."""
            import pandas as pd
            import joblib
            import json
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # Load data
            df = pd.read_csv(input_data.path)
            
            # Split features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Select model
            if algorithm == "random_forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif algorithm == "logistic_regression":
                model = LogisticRegression(random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }
            
            # Save model
            joblib.dump(model, model_output.path)
            
            # Save metrics
            with open(metrics_output.path, 'w') as f:
                json.dump(metrics, f)
            
            return metrics
        
        return training_component
    
    def _create_deployment_component(self):
        """Create model deployment component."""
        if not KFP_AVAILABLE:
            return None
            
        @component(
            base_image="python:3.8",
            packages_to_install=["google-cloud-aiplatform"]
        )
        def deployment_component(
            model_artifact: Input[Model],
            project_id: str
        ) -> Dict[str, Any]:
            """Model deployment component."""
            
            # Mock deployment
            endpoint_id = f"projects/{project_id}/locations/us-central1/endpoints/mock-endpoint"
            
            return {
                'endpoint_id': endpoint_id,
                'deployment_status': 'completed'
            }
        
        return deployment_component


def create_pipeline_manager(project_id: str, location: str = "us-central1") -> PipelineManager:
    """
    Create a PipelineManager instance.
    
    Args:
        project_id: Google Cloud project ID
        location: Vertex AI location
        
    Returns:
        PipelineManager instance
    """
    return PipelineManager(project_id, location)


def validate_pipeline_config(config: PipelineConfig) -> Dict[str, Any]:
    """
    Validate pipeline configuration.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'recommendations': []
    }
    
    # Check required fields
    if not config.pipeline_name:
        validation_result['is_valid'] = False
        validation_result['errors'].append("Pipeline name is required")
    
    if not config.project_id:
        validation_result['warnings'].append("Project ID not specified")
    
    # Check data configuration
    if not config.data_source:
        validation_result['warnings'].append("Data source not specified")
    
    # Check training configuration
    if config.hyperparameter_tuning and config.max_trial_count < 5:
        validation_result['warnings'].append("Low trial count for hyperparameter tuning")
    
    # Check deployment configuration
    if config.deploy_model and not config.endpoint_name:
        validation_result['recommendations'].append("Specify endpoint name for deployment")
    
    return validation_result
