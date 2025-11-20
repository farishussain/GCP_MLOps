"""
Simple Pipeline Orchestration Module

This module provides a simplified pipeline orchestration system that works
without external dependencies like KFP, focusing on local execution and
coordination of MLOps components.

Classes:
    SimplePipeline: Basic pipeline execution
    PipelineStep: Individual pipeline step
    LocalPipelineRunner: Local pipeline execution engine

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

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Pipeline step status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PipelineType(Enum):
    """Types of pipelines."""
    TRAINING = "training"
    DEPLOYMENT = "deployment"
    BATCH_PREDICTION = "batch_prediction"
    RETRAINING = "retraining"
    FULL_MLOPS = "full_mlops"


@dataclass
class PipelineStep:
    """Individual step in a pipeline."""
    name: str
    description: str
    function: Callable
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2


@dataclass
class SimplePipelineConfig:
    """Configuration for simple pipeline."""
    name: str
    description: str = ""
    pipeline_type: PipelineType = PipelineType.TRAINING
    
    # Input/output paths
    input_data_path: str = ""
    output_model_path: str = ""
    output_metrics_path: str = ""
    
    # Pipeline parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Execution settings
    fail_fast: bool = True
    enable_retries: bool = True
    parallel_execution: bool = False


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    pipeline_name: str
    status: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: float
    steps_completed: int
    steps_total: int
    success_rate: float
    outputs: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None


class SimplePipeline:
    """
    Simple pipeline for orchestrating MLOps steps.
    
    Provides basic pipeline functionality without external dependencies,
    focusing on local execution and step coordination.
    """
    
    def __init__(self, config: SimplePipelineConfig):
        """
        Initialize simple pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.steps: List[PipelineStep] = []
        self.context: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"pipeline.{config.name}")
    
    def add_step(self, step: PipelineStep) -> None:
        """
        Add a step to the pipeline.
        
        Args:
            step: PipelineStep to add
        """
        self.steps.append(step)
        self.logger.info(f"Added step: {step.name}")
    
    def execute(self) -> PipelineResult:
        """
        Execute the pipeline.
        
        Returns:
            PipelineResult object
        """
        self.logger.info(f"Starting pipeline: {self.config.name}")
        start_time = datetime.now()
        
        # Initialize context with parameters
        self.context.update(self.config.parameters)
        
        steps_completed = 0
        pipeline_outputs = {}
        pipeline_metrics = {}
        error_message = None
        
        try:
            for i, step in enumerate(self.steps):
                self.logger.info(f"Executing step {i+1}/{len(self.steps)}: {step.name}")
                
                success = self._execute_step(step)
                
                if success:
                    steps_completed += 1
                    # Merge step outputs into context
                    self.context.update(step.outputs)
                    pipeline_outputs.update(step.outputs)
                    
                    # Extract metrics if available
                    if 'metrics' in step.outputs:
                        pipeline_metrics.update(step.outputs['metrics'])
                else:
                    error_message = step.error_message
                    if self.config.fail_fast:
                        self.logger.error(f"Pipeline failed at step: {step.name}")
                        break
                    else:
                        self.logger.warning(f"Step failed but continuing: {step.name}")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            success_rate = steps_completed / len(self.steps) if self.steps else 0.0
            
            # Determine overall status
            if steps_completed == len(self.steps):
                status = "completed"
            elif steps_completed > 0:
                status = "partial"
            else:
                status = "failed"
            
            result = PipelineResult(
                pipeline_name=self.config.name,
                status=status,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                steps_completed=steps_completed,
                steps_total=len(self.steps),
                success_rate=success_rate,
                outputs=pipeline_outputs,
                metrics=pipeline_metrics,
                error_message=error_message
            )
            
            self.logger.info(f"Pipeline completed: {status} ({steps_completed}/{len(self.steps)} steps)")
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.error(f"Pipeline execution failed: {e}")
            
            return PipelineResult(
                pipeline_name=self.config.name,
                status="failed",
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                steps_completed=steps_completed,
                steps_total=len(self.steps),
                success_rate=0.0,
                error_message=str(e)
            )
    
    def _execute_step(self, step: PipelineStep) -> bool:
        """
        Execute a single pipeline step.
        
        Args:
            step: PipelineStep to execute
            
        Returns:
            True if step executed successfully
        """
        step.start_time = datetime.now()
        step.status = StepStatus.RUNNING
        
        for attempt in range(step.max_retries + 1):
            try:
                # Prepare step inputs from context
                step_inputs = {**step.inputs, **self.context}
                
                # Execute step function
                outputs = step.function(**step_inputs)
                
                # Store outputs
                if isinstance(outputs, dict):
                    step.outputs = outputs
                else:
                    step.outputs = {'result': outputs}
                
                step.status = StepStatus.COMPLETED
                step.end_time = datetime.now()
                
                self.logger.info(f"Step completed: {step.name}")
                return True
                
            except Exception as e:
                step.retry_count = attempt
                step.error_message = str(e)
                
                if attempt < step.max_retries and self.config.enable_retries:
                    self.logger.warning(f"Step failed, retrying ({attempt + 1}/{step.max_retries}): {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    step.status = StepStatus.FAILED
                    step.end_time = datetime.now()
                    self.logger.error(f"Step failed after {attempt + 1} attempts: {e}")
                    return False
        
        return False


class LocalPipelineRunner:
    """
    Local pipeline execution engine.
    
    Provides utilities for creating and running common MLOps pipelines
    without external orchestration systems.
    """
    
    def __init__(self):
        """Initialize local pipeline runner."""
        self.pipelines: Dict[str, SimplePipeline] = {}
        self.results: Dict[str, PipelineResult] = {}
    
    def create_training_pipeline(self, config: SimplePipelineConfig) -> SimplePipeline:
        """
        Create a model training pipeline.
        
        Args:
            config: Pipeline configuration
            
        Returns:
            SimplePipeline instance
        """
        config.pipeline_type = PipelineType.TRAINING
        pipeline = SimplePipeline(config)
        
        # Add data loading step
        data_loading_step = PipelineStep(
            name="data_loading",
            description="Load and validate input data",
            function=self._data_loading_function
        )
        pipeline.add_step(data_loading_step)
        
        # Add data preprocessing step
        preprocessing_step = PipelineStep(
            name="data_preprocessing",
            description="Preprocess and feature engineering",
            function=self._preprocessing_function
        )
        pipeline.add_step(preprocessing_step)
        
        # Add model training step
        training_step = PipelineStep(
            name="model_training",
            description="Train machine learning model",
            function=self._training_function
        )
        pipeline.add_step(training_step)
        
        # Add model evaluation step
        evaluation_step = PipelineStep(
            name="model_evaluation",
            description="Evaluate model performance",
            function=self._evaluation_function
        )
        pipeline.add_step(evaluation_step)
        
        # Add model saving step
        saving_step = PipelineStep(
            name="model_saving",
            description="Save trained model and artifacts",
            function=self._model_saving_function
        )
        pipeline.add_step(saving_step)
        
        self.pipelines[config.name] = pipeline
        return pipeline
    
    def create_deployment_pipeline(self, config: SimplePipelineConfig) -> SimplePipeline:
        """
        Create a model deployment pipeline.
        
        Args:
            config: Pipeline configuration
            
        Returns:
            SimplePipeline instance
        """
        config.pipeline_type = PipelineType.DEPLOYMENT
        pipeline = SimplePipeline(config)
        
        # Add model loading step
        model_loading_step = PipelineStep(
            name="model_loading",
            description="Load trained model",
            function=self._model_loading_function
        )
        pipeline.add_step(model_loading_step)
        
        # Add model validation step
        validation_step = PipelineStep(
            name="model_validation",
            description="Validate model for deployment",
            function=self._model_validation_function
        )
        pipeline.add_step(validation_step)
        
        # Add endpoint creation step
        endpoint_step = PipelineStep(
            name="endpoint_creation",
            description="Create deployment endpoint",
            function=self._endpoint_creation_function
        )
        pipeline.add_step(endpoint_step)
        
        # Add model deployment step
        deployment_step = PipelineStep(
            name="model_deployment",
            description="Deploy model to endpoint",
            function=self._deployment_function
        )
        pipeline.add_step(deployment_step)
        
        # Add deployment validation step
        deploy_validation_step = PipelineStep(
            name="deployment_validation",
            description="Validate deployment health",
            function=self._deployment_validation_function
        )
        pipeline.add_step(deploy_validation_step)
        
        self.pipelines[config.name] = pipeline
        return pipeline
    
    def create_full_mlops_pipeline(self, config: SimplePipelineConfig) -> SimplePipeline:
        """
        Create a full end-to-end MLOps pipeline.
        
        Args:
            config: Pipeline configuration
            
        Returns:
            SimplePipeline instance
        """
        config.pipeline_type = PipelineType.FULL_MLOPS
        pipeline = SimplePipeline(config)
        
        # Data pipeline steps
        pipeline.add_step(PipelineStep(
            name="data_ingestion",
            description="Ingest data from source",
            function=self._data_ingestion_function
        ))
        
        pipeline.add_step(PipelineStep(
            name="data_validation",
            description="Validate data quality",
            function=self._data_validation_function
        ))
        
        pipeline.add_step(PipelineStep(
            name="data_preprocessing",
            description="Preprocess and engineer features", 
            function=self._preprocessing_function
        ))
        
        # Training pipeline steps
        pipeline.add_step(PipelineStep(
            name="model_training",
            description="Train ML model",
            function=self._training_function
        ))
        
        pipeline.add_step(PipelineStep(
            name="model_evaluation",
            description="Evaluate model performance",
            function=self._evaluation_function
        ))
        
        pipeline.add_step(PipelineStep(
            name="model_validation",
            description="Validate model for production",
            function=self._model_validation_function
        ))
        
        # Deployment pipeline steps (conditional)
        if config.parameters.get('deploy_model', False):
            pipeline.add_step(PipelineStep(
                name="model_deployment",
                description="Deploy model to production",
                function=self._deployment_function
            ))
            
            pipeline.add_step(PipelineStep(
                name="deployment_monitoring",
                description="Set up deployment monitoring",
                function=self._monitoring_setup_function
            ))
        
        self.pipelines[config.name] = pipeline
        return pipeline
    
    def run_pipeline(self, pipeline_name: str) -> PipelineResult:
        """
        Run a pipeline.
        
        Args:
            pipeline_name: Name of pipeline to run
            
        Returns:
            PipelineResult object
        """
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_name} not found")
        
        pipeline = self.pipelines[pipeline_name]
        result = pipeline.execute()
        
        # Store result
        self.results[pipeline_name] = result
        
        return result
    
    def get_pipeline_result(self, pipeline_name: str) -> Optional[PipelineResult]:
        """
        Get pipeline execution result.
        
        Args:
            pipeline_name: Name of pipeline
            
        Returns:
            PipelineResult object or None
        """
        return self.results.get(pipeline_name)
    
    def list_pipelines(self) -> List[str]:
        """
        List all registered pipelines.
        
        Returns:
            List of pipeline names
        """
        return list(self.pipelines.keys())
    
    # Pipeline step functions
    def _data_loading_function(self, **kwargs) -> Dict[str, Any]:
        """Data loading step function."""
        logger.info("Executing data loading step")
        
        # Mock data loading
        import pandas as pd
        
        # Create sample data
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5] * 20,
            'feature2': [2, 4, 6, 8, 10] * 20,
            'feature3': [0.1, 0.2, 0.3, 0.4, 0.5] * 20,
            'target': [0, 1, 0, 1, 1] * 20
        })
        
        return {
            'data': data,
            'data_shape': list(data.shape),
            'feature_columns': ['feature1', 'feature2', 'feature3'],
            'target_column': 'target'
        }
    
    def _data_ingestion_function(self, **kwargs) -> Dict[str, Any]:
        """Data ingestion step function."""
        logger.info("Executing data ingestion step")
        return self._data_loading_function(**kwargs)
    
    def _data_validation_function(self, **kwargs) -> Dict[str, Any]:
        """Data validation step function."""
        logger.info("Executing data validation step")
        
        data = kwargs.get('data')
        if data is None:
            raise ValueError("No data provided for validation")
        
        # Basic validation checks
        validation_results = {
            'has_missing_values': data.isnull().any().any(),
            'num_rows': len(data),
            'num_columns': len(data.columns),
            'validation_passed': True
        }
        
        # Check minimum data requirements
        if len(data) < 10:
            validation_results['validation_passed'] = False
            raise ValueError("Insufficient data: less than 10 rows")
        
        return validation_results
    
    def _preprocessing_function(self, **kwargs) -> Dict[str, Any]:
        """Data preprocessing step function."""
        logger.info("Executing data preprocessing step")
        
        data = kwargs.get('data')
        if data is None:
            raise ValueError("No data provided for preprocessing")
        
        # Simple preprocessing
        from sklearn.preprocessing import StandardScaler
        
        feature_columns = kwargs.get('feature_columns', ['feature1', 'feature2', 'feature3'])
        target_column = kwargs.get('target_column', 'target')
        
        # Separate features and target
        X = data[feature_columns]
        y = data[target_column]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create processed dataframe
        import pandas as pd
        processed_data = pd.DataFrame(X_scaled, columns=feature_columns)
        processed_data[target_column] = y.values
        
        return {
            'processed_data': processed_data,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'target_column': target_column
        }
    
    def _training_function(self, **kwargs) -> Dict[str, Any]:
        """Model training step function."""
        logger.info("Executing model training step")
        
        # Get processed data
        data = kwargs.get('processed_data', kwargs.get('data'))
        if data is None:
            raise ValueError("No data provided for training")
        
        feature_columns = kwargs.get('feature_columns', ['feature1', 'feature2', 'feature3'])
        target_column = kwargs.get('target_column', 'target')
        algorithm = kwargs.get('algorithm', 'random_forest')
        
        # Prepare training data
        X = data[feature_columns]
        y = data[target_column]
        
        # Train model
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Select and train model
        if algorithm == 'random_forest':
            model = RandomForestClassifier(n_estimators=50, random_state=42)
        elif algorithm == 'logistic_regression':
            model = LogisticRegression(random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        model.fit(X_train, y_train)
        
        return {
            'trained_model': model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'algorithm': algorithm
        }
    
    def _evaluation_function(self, **kwargs) -> Dict[str, Any]:
        """Model evaluation step function."""
        logger.info("Executing model evaluation step")
        
        model = kwargs.get('trained_model')
        X_test = kwargs.get('X_test')
        y_test = kwargs.get('y_test')
        
        if model is None or X_test is None or y_test is None:
            raise ValueError("Missing model or test data for evaluation")
        
        # Evaluate model
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        predictions = model.predict(X_test)
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, predictions)),
            'precision': float(precision_score(y_test, predictions, average='weighted')),
            'recall': float(recall_score(y_test, predictions, average='weighted')),
            'f1_score': float(f1_score(y_test, predictions, average='weighted'))
        }
        
        return {
            'metrics': metrics,
            'predictions': predictions.tolist()
        }
    
    def _model_saving_function(self, **kwargs) -> Dict[str, Any]:
        """Model saving step function."""
        logger.info("Executing model saving step")
        
        model = kwargs.get('trained_model')
        output_path = kwargs.get('output_model_path', '/tmp/model.joblib')
        
        if model is None:
            raise ValueError("No trained model to save")
        
        # Save model
        import joblib
        joblib.dump(model, output_path)
        
        return {
            'model_path': output_path,
            'model_saved': True
        }
    
    def _model_loading_function(self, **kwargs) -> Dict[str, Any]:
        """Model loading step function."""
        logger.info("Executing model loading step")
        
        model_path = kwargs.get('model_path', kwargs.get('output_model_path'))
        
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"Model file not found: {model_path}")
        
        # Load model
        import joblib
        model = joblib.load(model_path)
        
        return {
            'loaded_model': model
        }
    
    def _model_validation_function(self, **kwargs) -> Dict[str, Any]:
        """Model validation step function."""
        logger.info("Executing model validation step")
        
        model = kwargs.get('trained_model', kwargs.get('loaded_model'))
        metrics = kwargs.get('metrics', {})
        
        if model is None:
            raise ValueError("No model provided for validation")
        
        # Validate model
        validation_passed = True
        validation_results = {
            'model_type': str(type(model)),
            'validation_passed': validation_passed
        }
        
        # Check model performance
        if metrics:
            accuracy = metrics.get('accuracy', 0.0)
            if accuracy < 0.6:  # Minimum accuracy threshold
                validation_passed = False
                validation_results['validation_passed'] = False
                validation_results['reason'] = f"Low accuracy: {accuracy}"
        
        return validation_results
    
    def _endpoint_creation_function(self, **kwargs) -> Dict[str, Any]:
        """Endpoint creation step function."""
        logger.info("Executing endpoint creation step")
        
        # Mock endpoint creation
        endpoint_id = f"endpoint-{int(time.time())}"
        
        return {
            'endpoint_id': endpoint_id,
            'endpoint_url': f"https://ml-api.example.com/{endpoint_id}"
        }
    
    def _deployment_function(self, **kwargs) -> Dict[str, Any]:
        """Model deployment step function."""
        logger.info("Executing model deployment step")
        
        model = kwargs.get('trained_model', kwargs.get('loaded_model'))
        endpoint_id = kwargs.get('endpoint_id')
        
        if model is None:
            raise ValueError("No model to deploy")
        
        # Mock deployment
        deployment_id = f"deployment-{int(time.time())}"
        
        return {
            'deployment_id': deployment_id,
            'endpoint_id': endpoint_id,
            'deployment_status': 'deployed'
        }
    
    def _deployment_validation_function(self, **kwargs) -> Dict[str, Any]:
        """Deployment validation step function."""
        logger.info("Executing deployment validation step")
        
        deployment_id = kwargs.get('deployment_id')
        endpoint_id = kwargs.get('endpoint_id')
        
        # Mock validation
        validation_results = {
            'deployment_healthy': True,
            'endpoint_accessible': True,
            'prediction_test_passed': True
        }
        
        return validation_results
    
    def _monitoring_setup_function(self, **kwargs) -> Dict[str, Any]:
        """Monitoring setup step function."""
        logger.info("Executing monitoring setup step")
        
        # Mock monitoring setup
        return {
            'monitoring_configured': True,
            'alerts_enabled': True,
            'dashboard_url': 'https://monitoring.example.com/dashboard'
        }


def create_pipeline_runner() -> LocalPipelineRunner:
    """
    Create a LocalPipelineRunner instance.
    
    Returns:
        LocalPipelineRunner instance
    """
    return LocalPipelineRunner()


def run_sample_pipeline() -> PipelineResult:
    """
    Run a sample MLOps pipeline for demonstration.
    
    Returns:
        PipelineResult object
    """
    # Create pipeline runner
    runner = LocalPipelineRunner()
    
    # Create sample training pipeline
    config = SimplePipelineConfig(
        name="sample_training_pipeline",
        description="Sample training pipeline for demonstration",
        parameters={
            'algorithm': 'random_forest',
            'output_model_path': '/tmp/sample_model.joblib'
        }
    )
    
    # Create and run pipeline
    pipeline = runner.create_training_pipeline(config)
    result = runner.run_pipeline(config.name)
    
    return result
