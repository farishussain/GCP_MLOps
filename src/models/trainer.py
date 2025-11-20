"""
Model training utilities for MLOps pipeline.

This module provides model training capabilities including:
- Multiple ML algorithms support
- Hyperparameter tuning
- Cross-validation
- Model persistence
- Performance evaluation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from pathlib import Path

from ..utils import save_json, timer

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Model training and evaluation utility class."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.best_model = None
        self.best_score = 0.0
        
        # Initialize models with default parameters
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize ML models with default parameters."""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                random_state=self.random_state
            ),
            'svm': SVC(
                kernel='rbf',
                random_state=self.random_state
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),
            'knn': KNeighborsClassifier(
                n_neighbors=5
            ),
            'naive_bayes': GaussianNB(),
            'decision_tree': DecisionTreeClassifier(
                random_state=self.random_state
            )
        }
        
        logger.info(f"Initialized {len(self.models)} models")
    
    @timer
    def train_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Train a specific model.
        
        Args:
            model_name: Name of the model to train.
            X_train: Training features.
            y_train: Training targets.
            hyperparameters: Optional hyperparameters to override defaults.
            
        Returns:
            Trained model instance.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available. "
                           f"Choose from: {list(self.models.keys())}")
        
        logger.info(f"Training model: {model_name}")
        
        # Get model instance
        model = self.models[model_name]
        
        # Set hyperparameters if provided
        if hyperparameters:
            model.set_params(**hyperparameters)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Store trained model
        self.trained_models[model_name] = model
        
        logger.info(f"Model {model_name} trained successfully")
        return model
    
    @timer
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict[str, Any]:
        """
        Train all available models.
        
        Args:
            X_train: Training features.
            y_train: Training targets.
            
        Returns:
            Dictionary of trained models.
        """
        logger.info("Training all models...")
        
        for model_name in self.models.keys():
            try:
                self.train_model(model_name, X_train, y_train)
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
        
        logger.info(f"Trained {len(self.trained_models)} models successfully")
        return self.trained_models
    
    def evaluate_model(
        self,
        model_name: str,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model.
        
        Args:
            model_name: Name of the model to evaluate.
            X_test: Test features.
            y_test: Test targets.
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        evaluation = {
            'model_name': model_name,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist()
        }
        
        logger.info(f"Model {model_name} - Accuracy: {accuracy:.4f}")
        
        return evaluation
    
    def evaluate_all_models(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all trained models.
        
        Args:
            X_test: Test features.
            y_test: Test targets.
            
        Returns:
            Dictionary containing evaluation results for all models.
        """
        logger.info("Evaluating all models...")
        
        evaluations = {}
        
        for model_name in self.trained_models.keys():
            try:
                evaluations[model_name] = self.evaluate_model(model_name, X_test, y_test)
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
        
        # Find best model
        best_accuracy = 0.0
        best_model_name = None
        
        for model_name, evaluation in evaluations.items():
            if evaluation['accuracy'] > best_accuracy:
                best_accuracy = evaluation['accuracy']
                best_model_name = model_name
        
        if best_model_name:
            self.best_model = self.trained_models[best_model_name]
            self.best_score = best_accuracy
            logger.info(f"Best model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        
        return evaluations
    
    @timer
    def hyperparameter_tuning(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Dict[str, List[Any]],
        cv: int = 5,
        scoring: str = 'accuracy'
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for a model.
        
        Args:
            model_name: Name of the model to tune.
            X_train: Training features.
            y_train: Training targets.
            param_grid: Parameter grid for tuning.
            cv: Number of cross-validation folds.
            scoring: Scoring metric.
            
        Returns:
            Dictionary containing tuning results.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        logger.info(f"Starting hyperparameter tuning for {model_name}")
        
        model = self.models[model_name]
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update trained model with best parameters
        self.trained_models[model_name] = grid_search.best_estimator_
        
        results = {
            'model_name': model_name,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"Hyperparameter tuning completed for {model_name}. "
                   f"Best score: {grid_search.best_score_:.4f}")
        
        return results
    
    def cross_validate_model(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: str = 'accuracy'
    ) -> Dict[str, Any]:
        """
        Perform cross-validation for a model.
        
        Args:
            model_name: Name of the model.
            X: Features.
            y: Targets.
            cv: Number of cross-validation folds.
            scoring: Scoring metric.
            
        Returns:
            Dictionary containing cross-validation results.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        model = self.models[model_name]
        
        logger.info(f"Performing {cv}-fold cross-validation for {model_name}")
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        results = {
            'model_name': model_name,
            'cv_scores': scores.tolist(),
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'cv_folds': cv
        }
        
        logger.info(f"Cross-validation completed for {model_name}. "
                   f"Mean score: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
        
        return results
    
    def save_model(self, model_name: str, output_path: str) -> None:
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model to save.
            output_path: Path to save the model.
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        model = self.trained_models[model_name]
        joblib.dump(model, output_path)
        
        logger.info(f"Model {model_name} saved to {output_path}")
    
    def load_model(self, model_name: str, model_path: str) -> Any:
        """
        Load a saved model from disk.
        
        Args:
            model_name: Name to assign to the loaded model.
            model_path: Path to the saved model.
            
        Returns:
            Loaded model instance.
        """
        model = joblib.load(model_path)
        self.trained_models[model_name] = model
        
        logger.info(f"Model loaded from {model_path} as {model_name}")
        
        return model
    
    def get_feature_importance(self, model_name: str) -> Optional[Dict[str, float]]:
        """
        Get feature importance for tree-based models.
        
        Args:
            model_name: Name of the model.
            
        Returns:
            Dictionary of feature importances or None if not available.
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            return {
                f'feature_{i}': importance 
                for i, importance in enumerate(model.feature_importances_)
            }
        else:
            logger.warning(f"Model {model_name} does not support feature importance")
            return None
    
    def save_evaluation_results(
        self,
        evaluations: Dict[str, Dict[str, Any]],
        output_path: str
    ) -> None:
        """
        Save evaluation results to JSON file.
        
        Args:
            evaluations: Evaluation results dictionary.
            output_path: Path to save the results.
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_evaluations = {}
        
        for model_name, evaluation in evaluations.items():
            serializable_evaluations[model_name] = {
                'model_name': evaluation['model_name'],
                'accuracy': float(evaluation['accuracy']),
                'classification_report': evaluation['classification_report'],
                'confusion_matrix': evaluation['confusion_matrix'],
                'predictions': evaluation['predictions']
            }
        
        save_json(serializable_evaluations, output_path)
        logger.info(f"Evaluation results saved to {output_path}")

def get_default_param_grids() -> Dict[str, Dict[str, List[Any]]]:
    """
    Get default parameter grids for hyperparameter tuning.
    
    Returns:
        Dictionary of parameter grids for each model.
    """
    return {
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10]
        },
        'logistic_regression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs']
        },
        'svm': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        },
        'gradient_boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'knn': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }
    }
