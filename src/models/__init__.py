"""
Models Package

This package provides comprehensive machine learning model training, evaluation,
and deployment capabilities for the MLOps pipeline.

Modules:
    trainer: Model training with multiple algorithms and hyperparameter tuning
    evaluator: Model evaluation, visualization, and performance analysis
    model_deployment: Model deployment and serving utilities
    model_registry: Model versioning and registry management

Key Classes:
    ModelTrainer: Train multiple ML models with automated hyperparameter tuning
    ModelEvaluator: Comprehensive model evaluation and visualization
    EvaluationResults: Container for model performance metrics
    ModelConfig: Configuration for model training parameters

Example Usage:
    from src.models.trainer import ModelTrainer, ModelConfig
    from src.models.evaluator import ModelEvaluator
    
    # Load data
    trainer = ModelTrainer()
    trainer.load_data(X_train, X_test, y_train, y_test)
    
    # Train models
    results = trainer.train_all_models()
    
    # Evaluate and visualize
    evaluator = ModelEvaluator()
    evaluator.create_evaluation_report(results)

Author: MLOps Team
Version: 1.0.0
"""

from .trainer import ModelTrainer, ModelConfig, EvaluationResults
from .evaluator import ModelEvaluator, create_model_evaluation_dashboard
from .model_registry import ModelRegistry

__all__ = [
    'ModelTrainer',
    'ModelConfig', 
    'EvaluationResults',
    'ModelEvaluator',
    'create_model_evaluation_dashboard',
    'ModelRegistry'
]

__version__ = '1.0.0'
__author__ = 'MLOps Team'
