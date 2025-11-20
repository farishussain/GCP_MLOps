"""
Model Training Module

This module provides a unified interface for training multiple machine learning models
with comprehensive evaluation, hyperparameter tuning, and model comparison capabilities.

Classes:
    ModelTrainer: Main class for training and evaluating ML models
    ModelConfig: Configuration dataclass for model parameters
    EvaluationResults: Dataclass for storing evaluation metrics and results

Author: MLOps Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
import joblib
import json
from pathlib import Path
import logging
from datetime import datetime
import warnings

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Model selection and evaluation
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve
)

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model training parameters."""
    name: str
    algorithm: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    hyperparameter_grid: Dict[str, List[Any]] = field(default_factory=dict)
    cross_validation_folds: int = 5
    random_state: int = 42
    enable_hyperparameter_tuning: bool = True
    scoring_metric: str = 'accuracy'


@dataclass
class EvaluationResults:
    """Container for model evaluation results."""
    model_name: str
    algorithm: str
    train_accuracy: float
    test_accuracy: float
    cross_val_mean: float
    cross_val_std: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[str] = None
    best_parameters: Optional[Dict[str, Any]] = None
    training_time: float = 0.0
    prediction_time: float = 0.0
    feature_importance: Optional[Dict[str, float]] = None


class ModelTrainer:
    """
    Comprehensive machine learning model trainer with support for multiple algorithms,
    hyperparameter tuning, and extensive evaluation metrics.
    
    Features:
    - Multiple ML algorithms (RandomForest, SVM, LogisticRegression, etc.)
    - Automated hyperparameter tuning with GridSearchCV
    - Comprehensive evaluation metrics and visualizations
    - Model persistence and metadata tracking
    - Cross-validation with stratified folds
    - Feature importance analysis
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the ModelTrainer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, EvaluationResults] = {}
        self.trained_models: Dict[str, Any] = {}
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        
        logger.info(f"ModelTrainer initialized with random_state={random_state}")
    
    def load_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                  y_train: pd.Series, y_test: pd.Series) -> None:
        """
        Load training and testing data.
        
        Args:
            X_train: Training features
            X_test: Testing features  
            y_train: Training targets
            y_test: Testing targets
        """
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        self.y_train = y_train.copy()
        self.y_test = y_test.copy()
        
        logger.info(f"Data loaded: Train shape {X_train.shape}, Test shape {X_test.shape}")
        logger.info(f"Target classes: {sorted(y_train.unique())}")
    
    def get_default_models(self) -> Dict[str, ModelConfig]:
        """
        Get default model configurations for common ML algorithms.
        
        Returns:
            Dictionary of model configurations
        """
        models = {
            'random_forest': ModelConfig(
                name='Random Forest',
                algorithm='RandomForestClassifier',
                parameters={'random_state': self.random_state},
                hyperparameter_grid={
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            ),
            'logistic_regression': ModelConfig(
                name='Logistic Regression',
                algorithm='LogisticRegression',
                parameters={'random_state': self.random_state, 'max_iter': 1000},
                hyperparameter_grid={
                    'C': [0.1, 1.0, 10.0],
                    'solver': ['liblinear', 'lbfgs']
                }
            ),
            'svm': ModelConfig(
                name='Support Vector Machine',
                algorithm='SVC',
                parameters={'random_state': self.random_state, 'probability': True},
                hyperparameter_grid={
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }
            ),
            'gradient_boosting': ModelConfig(
                name='Gradient Boosting',
                algorithm='GradientBoostingClassifier',
                parameters={'random_state': self.random_state},
                hyperparameter_grid={
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            ),
            'knn': ModelConfig(
                name='K-Nearest Neighbors',
                algorithm='KNeighborsClassifier',
                parameters={},
                hyperparameter_grid={
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            ),
            'naive_bayes': ModelConfig(
                name='Gaussian Naive Bayes',
                algorithm='GaussianNB',
                parameters={},
                hyperparameter_grid={}
            ),
            'decision_tree': ModelConfig(
                name='Decision Tree',
                algorithm='DecisionTreeClassifier',
                parameters={'random_state': self.random_state},
                hyperparameter_grid={
                    'max_depth': [None, 5, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            )
        }
        
        return models
    
    def create_model_instance(self, config: ModelConfig) -> Any:
        """
        Create a model instance from configuration.
        
        Args:
            config: Model configuration
            
        Returns:
            Instantiated model object
        """
        model_classes = {
            'RandomForestClassifier': RandomForestClassifier,
            'LogisticRegression': LogisticRegression,
            'SVC': SVC,
            'GradientBoostingClassifier': GradientBoostingClassifier,
            'KNeighborsClassifier': KNeighborsClassifier,
            'GaussianNB': GaussianNB,
            'DecisionTreeClassifier': DecisionTreeClassifier
        }
        
        if config.algorithm not in model_classes:
            raise ValueError(f"Unsupported algorithm: {config.algorithm}")
        
        model_class = model_classes[config.algorithm]
        return model_class(**config.parameters)
    
    def train_model(self, config: ModelConfig) -> EvaluationResults:
        """
        Train a single model with optional hyperparameter tuning.
        
        Args:
            config: Model configuration
            
        Returns:
            Evaluation results
        """
        if self.X_train is None or self.y_train is None or self.X_test is None or self.y_test is None:
            raise ValueError("No training data loaded. Call load_data() first.")
        
        start_time = datetime.now()
        logger.info(f"Training {config.name}...")
        
        # Create base model
        model = self.create_model_instance(config)
        
        # Hyperparameter tuning if enabled and grid provided
        if config.enable_hyperparameter_tuning and config.hyperparameter_grid:
            logger.info(f"Performing hyperparameter tuning for {config.name}")
            
            cv = StratifiedKFold(n_splits=config.cross_validation_folds, 
                               shuffle=True, random_state=self.random_state)
            
            grid_search = GridSearchCV(
                model,
                config.hyperparameter_grid,
                cv=cv,
                scoring=config.scoring_metric,
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(self.X_train, self.y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            logger.info(f"Best parameters for {config.name}: {best_params}")
        else:
            best_model = model
            best_params = None
            best_model.fit(self.X_train, self.y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Make predictions
        pred_start = datetime.now()
        train_pred = best_model.predict(self.X_train)
        test_pred = best_model.predict(self.X_test)
        prediction_time = (datetime.now() - pred_start).total_seconds()
        
        # Cross-validation
        cv_scores = cross_val_score(
            best_model, self.X_train, self.y_train,
            cv=StratifiedKFold(n_splits=config.cross_validation_folds, 
                             shuffle=True, random_state=self.random_state),
            scoring=config.scoring_metric
        )
        
        # Calculate metrics
        train_accuracy = float(accuracy_score(self.y_train, train_pred))
        test_accuracy = float(accuracy_score(self.y_test, test_pred))
        precision = float(precision_score(self.y_test, test_pred, average='weighted'))
        recall = float(recall_score(self.y_test, test_pred, average='weighted'))
        f1 = float(f1_score(self.y_test, test_pred, average='weighted'))
        
        # ROC AUC (only for binary classification or models with predict_proba)
        roc_auc = None
        try:
            y_test_array = np.asarray(self.y_test)
            if hasattr(best_model, 'predict_proba') and len(np.unique(y_test_array)) == 2:
                test_proba = best_model.predict_proba(self.X_test)[:, 1]
                roc_auc = float(roc_auc_score(self.y_test, test_proba))
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC for {config.name}: {e}")
        
        # Confusion matrix and classification report
        conf_matrix = confusion_matrix(self.y_test, test_pred)
        class_report = str(classification_report(self.y_test, test_pred))
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(best_model, 'feature_importances_'):
            feature_names = self.X_train.columns.tolist()
            importances = best_model.feature_importances_
            feature_importance = dict(zip(feature_names, importances))
        elif hasattr(best_model, 'coef_') and len(best_model.coef_.shape) == 2:
            feature_names = self.X_train.columns.tolist()
            # For multi-class, use mean absolute coefficient
            importances = np.mean(np.abs(best_model.coef_), axis=0)
            feature_importance = dict(zip(feature_names, importances))
        
        # Store the trained model
        self.trained_models[config.name] = best_model
        
        # Create results
        results = EvaluationResults(
            model_name=config.name,
            algorithm=config.algorithm,
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy,
            cross_val_mean=float(cv_scores.mean()),
            cross_val_std=float(cv_scores.std()),
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            confusion_matrix=conf_matrix,
            classification_report=class_report,
            best_parameters=best_params,
            training_time=training_time,
            prediction_time=prediction_time,
            feature_importance=feature_importance
        )
        
        self.results[config.name] = results
        
        logger.info(f"Completed training {config.name} - Test Accuracy: {test_accuracy:.4f}")
        return results
    
    def train_all_models(self, custom_configs: Optional[Dict[str, ModelConfig]] = None) -> Dict[str, EvaluationResults]:
        """
        Train multiple models and compare performance.
        
        Args:
            custom_configs: Optional custom model configurations
            
        Returns:
            Dictionary of evaluation results for each model
        """
        configs = custom_configs if custom_configs else self.get_default_models()
        
        logger.info(f"Training {len(configs)} models...")
        
        results = {}
        for model_key, config in configs.items():
            try:
                result = self.train_model(config)
                results[config.name] = result
            except Exception as e:
                logger.error(f"Failed to train {config.name}: {e}")
                continue
        
        logger.info(f"Completed training {len(results)} models successfully")
        return results
    
    def get_model_comparison(self) -> pd.DataFrame:
        """
        Get a comparison dataframe of all trained models.
        
        Returns:
            DataFrame with model performance metrics
        """
        if not self.results:
            raise ValueError("No models trained yet. Call train_all_models() first.")
        
        comparison_data = []
        for name, result in self.results.items():
            comparison_data.append({
                'Model': result.model_name,
                'Algorithm': result.algorithm,
                'Test Accuracy': result.test_accuracy,
                'Cross-Val Mean': result.cross_val_mean,
                'Cross-Val Std': result.cross_val_std,
                'Precision': result.precision,
                'Recall': result.recall,
                'F1-Score': result.f1_score,
                'ROC AUC': result.roc_auc,
                'Training Time (s)': result.training_time,
                'Prediction Time (s)': result.prediction_time
            })
        
        df = pd.DataFrame(comparison_data)
        # Sort by test accuracy descending
        df = df.sort_values('Test Accuracy', ascending=False).reset_index(drop=True)
        
        return df
    
    def get_best_model(self, metric: str = 'test_accuracy') -> Tuple[str, EvaluationResults, Any]:
        """
        Get the best performing model based on specified metric.
        
        Args:
            metric: Metric to use for selection ('test_accuracy', 'cross_val_mean', 'f1_score', etc.)
            
        Returns:
            Tuple of (model_name, evaluation_results, trained_model)
        """
        if not self.results:
            raise ValueError("No models trained yet. Call train_all_models() first.")
        
        best_score = -np.inf
        best_name = None
        
        for name, result in self.results.items():
            score = getattr(result, metric, None)
            if score is not None and score > best_score:
                best_score = score
                best_name = name
        
        if best_name is None:
            raise ValueError(f"No models found with metric: {metric}")
        
        return best_name, self.results[best_name], self.trained_models[best_name]
    
    def save_model(self, model_name: str, filepath: Union[str, Path]) -> None:
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model to save
            filepath: Path to save the model file
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.trained_models.keys())}")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.trained_models[model_name],
            'results': self.results[model_name],
            'feature_names': self.X_train.columns.tolist() if self.X_train is not None else None,
            'target_classes': sorted(self.y_train.unique()) if self.y_train is not None else None,
            'saved_at': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model {model_name} saved to {filepath}")
    
    def save_results_summary(self, filepath: Union[str, Path]) -> None:
        """
        Save training results summary to JSON file.
        
        Args:
            filepath: Path to save the results file
        """
        if not self.results:
            raise ValueError("No results to save. Train models first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        summary = {
            'training_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_models': len(self.results),
                'data_shape': {
                    'train_samples': len(self.X_train) if self.X_train is not None else None,
                    'test_samples': len(self.X_test) if self.X_test is not None else None,
                    'features': len(self.X_train.columns) if self.X_train is not None else None
                }
            },
            'results': {}
        }
        
        for name, result in self.results.items():
            summary['results'][name] = {
                'model_name': result.model_name,
                'algorithm': result.algorithm,
                'test_accuracy': float(result.test_accuracy),
                'cross_val_mean': float(result.cross_val_mean),
                'cross_val_std': float(result.cross_val_std),
                'precision': float(result.precision),
                'recall': float(result.recall),
                'f1_score': float(result.f1_score),
                'roc_auc': float(result.roc_auc) if result.roc_auc else None,
                'training_time': float(result.training_time),
                'prediction_time': float(result.prediction_time),
                'best_parameters': result.best_parameters,
                'feature_importance': result.feature_importance
            }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results summary saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> Tuple[Any, EvaluationResults]:
        """
        Load a saved model from disk.
        
        Args:
            filepath: Path to the saved model file
            
        Returns:
            Tuple of (model, evaluation_results)
        """
        model_data = joblib.load(filepath)
        return model_data['model'], model_data['results']
