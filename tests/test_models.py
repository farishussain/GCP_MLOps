"""
Tests for the models package

This module contains comprehensive unit tests for all model training and 
evaluation functionality.

Test Categories:
- ModelTrainer functionality
- Model configuration and validation
- Evaluation results and metrics
- Model comparison and selection
- Error handling and edge cases

Author: MLOps Team
Version: 1.0.0
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Import modules to test
from src.models.trainer import ModelTrainer, ModelConfig, EvaluationResults
from src.models.evaluator import ModelEvaluator


class TestModelConfig:
    """Test the ModelConfig dataclass."""
    
    def test_model_config_creation(self):
        """Test basic model configuration creation."""
        config = ModelConfig(
            name="Test Model",
            algorithm="RandomForestClassifier"
        )
        
        assert config.name == "Test Model"
        assert config.algorithm == "RandomForestClassifier"
        assert config.random_state == 42
        assert config.enable_hyperparameter_tuning is True
        
    def test_model_config_with_parameters(self):
        """Test model configuration with custom parameters."""
        params = {'n_estimators': 100, 'max_depth': 10}
        hp_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
        
        config = ModelConfig(
            name="Custom Model",
            algorithm="RandomForestClassifier",
            parameters=params,
            hyperparameter_grid=hp_grid,
            random_state=123
        )
        
        assert config.parameters == params
        assert config.hyperparameter_grid == hp_grid
        assert config.random_state == 123


class TestEvaluationResults:
    """Test the EvaluationResults dataclass."""
    
    def test_evaluation_results_creation(self):
        """Test basic evaluation results creation."""
        results = EvaluationResults(
            model_name="Test Model",
            algorithm="RandomForestClassifier",
            train_accuracy=0.95,
            test_accuracy=0.90,
            cross_val_mean=0.88,
            cross_val_std=0.02,
            precision=0.89,
            recall=0.91,
            f1_score=0.90
        )
        
        assert results.model_name == "Test Model"
        assert results.test_accuracy == 0.90
        assert results.cross_val_mean == 0.88
        assert results.roc_auc is None  # Default value
        
    def test_evaluation_results_with_optional_fields(self):
        """Test evaluation results with optional fields."""
        conf_matrix = np.array([[10, 2], [1, 15]])
        feature_importance = {'feature1': 0.6, 'feature2': 0.4}
        
        results = EvaluationResults(
            model_name="Complete Model",
            algorithm="SVC",
            train_accuracy=0.95,
            test_accuracy=0.90,
            cross_val_mean=0.88,
            cross_val_std=0.02,
            precision=0.89,
            recall=0.91,
            f1_score=0.90,
            roc_auc=0.92,
            confusion_matrix=conf_matrix,
            feature_importance=feature_importance
        )
        
        assert results.roc_auc == 0.92
        np.testing.assert_array_equal(results.confusion_matrix, conf_matrix)
        assert results.feature_importance == feature_importance


class TestModelTrainer:
    """Test the ModelTrainer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample classification data for testing."""
        X, y = make_classification(
            n_samples=100, n_features=4, n_classes=3,
            n_informative=3, n_redundant=1,
            random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Convert to DataFrames/Series
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        y_train_series = pd.Series(y_train)
        y_test_series = pd.Series(y_test)
        
        return X_train_df, X_test_df, y_train_series, y_test_series
    
    def test_model_trainer_initialization(self):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(random_state=123)
        
        assert trainer.random_state == 123
        assert trainer.models == {}
        assert trainer.results == {}
        assert trainer.trained_models == {}
        assert trainer.X_train is None
        
    def test_load_data(self, sample_data):
        """Test data loading functionality."""
        X_train, X_test, y_train, y_test = sample_data
        trainer = ModelTrainer()
        
        trainer.load_data(X_train, X_test, y_train, y_test)
        
        assert trainer.X_train is not None
        assert trainer.X_test is not None
        assert trainer.y_train is not None
        assert trainer.y_test is not None
        assert trainer.X_train.shape[0] == len(y_train)
        assert trainer.X_test.shape[0] == len(y_test)
        
    def test_get_default_models(self):
        """Test default model configurations."""
        trainer = ModelTrainer()
        models = trainer.get_default_models()
        
        assert 'random_forest' in models
        assert 'logistic_regression' in models
        assert 'svm' in models
        
        # Check random forest config
        rf_config = models['random_forest']
        assert rf_config.name == 'Random Forest'
        assert rf_config.algorithm == 'RandomForestClassifier'
        assert 'n_estimators' in rf_config.hyperparameter_grid
        
    def test_create_model_instance(self):
        """Test model instance creation."""
        trainer = ModelTrainer()
        
        config = ModelConfig(
            name="Test RF",
            algorithm="RandomForestClassifier",
            parameters={'random_state': 42}
        )
        
        model = trainer.create_model_instance(config)
        
        # Check it's the right type
        from sklearn.ensemble import RandomForestClassifier
        assert isinstance(model, RandomForestClassifier)
        assert hasattr(model, 'random_state')
        
    def test_create_model_instance_unsupported(self):
        """Test error handling for unsupported algorithms."""
        trainer = ModelTrainer()
        
        config = ModelConfig(
            name="Unsupported",
            algorithm="UnsupportedClassifier"
        )
        
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            trainer.create_model_instance(config)
    
    def test_train_model_without_data(self):
        """Test training without loaded data raises error."""
        trainer = ModelTrainer()
        config = ModelConfig(name="Test", algorithm="RandomForestClassifier")
        
        with pytest.raises(ValueError, match="No training data loaded"):
            trainer.train_model(config)
    
    def test_train_single_model(self, sample_data):
        """Test training a single model."""
        X_train, X_test, y_train, y_test = sample_data
        trainer = ModelTrainer()
        trainer.load_data(X_train, X_test, y_train, y_test)
        
        config = ModelConfig(
            name="Test Random Forest",
            algorithm="RandomForestClassifier",
            parameters={'random_state': 42, 'n_estimators': 10},
            hyperparameter_grid={},  # Disable hyperparameter tuning for speed
            enable_hyperparameter_tuning=False
        )
        
        result = trainer.train_model(config)
        
        assert result.model_name == "Test Random Forest"
        assert result.algorithm == "RandomForestClassifier"
        assert 0 <= result.test_accuracy <= 1
        assert 0 <= result.train_accuracy <= 1
        assert result.training_time > 0
        assert "Test Random Forest" in trainer.trained_models
        
    def test_train_model_with_hyperparameter_tuning(self, sample_data):
        """Test training with hyperparameter tuning."""
        X_train, X_test, y_train, y_test = sample_data
        trainer = ModelTrainer()
        trainer.load_data(X_train, X_test, y_train, y_test)
        
        config = ModelConfig(
            name="Tuned RF",
            algorithm="RandomForestClassifier",
            parameters={'random_state': 42},
            hyperparameter_grid={
                'n_estimators': [5, 10],
                'max_depth': [3, 5]
            },
            enable_hyperparameter_tuning=True,
            cross_validation_folds=3  # Reduce for speed
        )
        
        result = trainer.train_model(config)
        
        assert result.model_name == "Tuned RF"
        assert result.best_parameters is not None
        assert 'n_estimators' in result.best_parameters
        assert 'max_depth' in result.best_parameters
        
    def test_train_all_models(self, sample_data):
        """Test training multiple models."""
        X_train, X_test, y_train, y_test = sample_data
        trainer = ModelTrainer()
        trainer.load_data(X_train, X_test, y_train, y_test)
        
        # Create custom configs for faster testing
        custom_configs = {
            'rf': ModelConfig(
                name="Fast RF",
                algorithm="RandomForestClassifier",
                parameters={'random_state': 42, 'n_estimators': 10},
                enable_hyperparameter_tuning=False
            ),
            'lr': ModelConfig(
                name="Fast LR",
                algorithm="LogisticRegression",
                parameters={'random_state': 42, 'max_iter': 100},
                enable_hyperparameter_tuning=False
            )
        }
        
        results = trainer.train_all_models(custom_configs)
        
        assert len(results) == 2
        assert "Fast RF" in results
        assert "Fast LR" in results
        
        # Check all models were stored
        assert "Fast RF" in trainer.trained_models
        assert "Fast LR" in trainer.trained_models
        
    def test_get_model_comparison(self, sample_data):
        """Test model comparison dataframe creation."""
        X_train, X_test, y_train, y_test = sample_data
        trainer = ModelTrainer()
        trainer.load_data(X_train, X_test, y_train, y_test)
        
        # Train some models first
        custom_configs = {
            'rf': ModelConfig(
                name="RF Model",
                algorithm="RandomForestClassifier",
                parameters={'random_state': 42, 'n_estimators': 10},
                enable_hyperparameter_tuning=False
            )
        }
        trainer.train_all_models(custom_configs)
        
        comparison_df = trainer.get_model_comparison()
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 1
        assert 'Model' in comparison_df.columns
        assert 'Test Accuracy' in comparison_df.columns
        assert comparison_df.iloc[0]['Model'] == "RF Model"
        
    def test_get_best_model(self, sample_data):
        """Test getting the best performing model."""
        X_train, X_test, y_train, y_test = sample_data
        trainer = ModelTrainer()
        trainer.load_data(X_train, X_test, y_train, y_test)
        
        # Train models
        custom_configs = {
            'rf': ModelConfig(
                name="RF Model",
                algorithm="RandomForestClassifier",
                parameters={'random_state': 42, 'n_estimators': 10},
                enable_hyperparameter_tuning=False
            ),
            'lr': ModelConfig(
                name="LR Model", 
                algorithm="LogisticRegression",
                parameters={'random_state': 42, 'max_iter': 100},
                enable_hyperparameter_tuning=False
            )
        }
        trainer.train_all_models(custom_configs)
        
        best_name, best_results, best_model = trainer.get_best_model()
        
        assert best_name in ["RF Model", "LR Model"]
        assert isinstance(best_results, EvaluationResults)
        assert best_model is not None
        
    def test_get_best_model_no_results(self):
        """Test error when no models trained."""
        trainer = ModelTrainer()
        
        with pytest.raises(ValueError, match="No models trained"):
            trainer.get_best_model()
            
    def test_save_and_load_model(self, sample_data):
        """Test model saving and loading."""
        X_train, X_test, y_train, y_test = sample_data
        trainer = ModelTrainer()
        trainer.load_data(X_train, X_test, y_train, y_test)
        
        # Train a model
        config = ModelConfig(
            name="Save Test",
            algorithm="RandomForestClassifier",
            parameters={'random_state': 42, 'n_estimators': 10},
            enable_hyperparameter_tuning=False
        )
        trainer.train_model(config)
        
        # Save model
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.pkl"
            trainer.save_model("Save Test", model_path)
            
            assert model_path.exists()
            
            # Load model back
            loaded_model, loaded_results = ModelTrainer.load_model(model_path)
            
            assert loaded_results.model_name == "Save Test"
            assert loaded_model is not None
            
            # Test that loaded model can make predictions
            predictions = loaded_model.predict(X_test)
            assert len(predictions) == len(y_test)
    
    def test_save_model_not_found(self, sample_data):
        """Test error when trying to save non-existent model."""
        X_train, X_test, y_train, y_test = sample_data
        trainer = ModelTrainer()
        trainer.load_data(X_train, X_test, y_train, y_test)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.pkl"
            
            with pytest.raises(ValueError, match="Model .* not found"):
                trainer.save_model("Nonexistent Model", model_path)
                
    def test_save_results_summary(self, sample_data):
        """Test saving results summary to JSON."""
        X_train, X_test, y_train, y_test = sample_data
        trainer = ModelTrainer()
        trainer.load_data(X_train, X_test, y_train, y_test)
        
        # Train a model
        config = ModelConfig(
            name="Summary Test",
            algorithm="RandomForestClassifier",
            parameters={'random_state': 42, 'n_estimators': 10},
            enable_hyperparameter_tuning=False
        )
        trainer.train_model(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = Path(temp_dir) / "results.json"
            trainer.save_results_summary(summary_path)
            
            assert summary_path.exists()
            
            # Load and check content
            import json
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            assert 'training_summary' in summary
            assert 'results' in summary
            assert 'Summary Test' in summary['results']


class TestModelEvaluator:
    """Test the ModelEvaluator class."""
    
    @pytest.fixture
    def sample_evaluation_results(self):
        """Create sample evaluation results for testing."""
        results = {
            'Random Forest': EvaluationResults(
                model_name="Random Forest",
                algorithm="RandomForestClassifier",
                train_accuracy=0.95,
                test_accuracy=0.90,
                cross_val_mean=0.88,
                cross_val_std=0.02,
                precision=0.89,
                recall=0.91,
                f1_score=0.90,
                feature_importance={'feature_0': 0.4, 'feature_1': 0.3, 'feature_2': 0.3}
            ),
            'Logistic Regression': EvaluationResults(
                model_name="Logistic Regression",
                algorithm="LogisticRegression",
                train_accuracy=0.88,
                test_accuracy=0.85,
                cross_val_mean=0.83,
                cross_val_std=0.03,
                precision=0.84,
                recall=0.86,
                f1_score=0.85
            )
        }
        return results
    
    def test_evaluator_initialization(self):
        """Test ModelEvaluator initialization."""
        evaluator = ModelEvaluator(figsize=(10, 6))
        
        assert evaluator.figsize == (10, 6)
        
    def test_plot_confusion_matrix(self):
        """Test confusion matrix plotting."""
        evaluator = ModelEvaluator()
        
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        
        # Test without saving
        fig = evaluator.plot_confusion_matrix(y_true, y_pred)
        assert fig is not None
        
        # Test with class names
        fig = evaluator.plot_confusion_matrix(
            y_true, y_pred, 
            class_names=['Class A', 'Class B', 'Class C']
        )
        assert fig is not None
        
    def test_plot_feature_importance(self):
        """Test feature importance plotting."""
        evaluator = ModelEvaluator()
        
        feature_importance = {
            'feature_1': 0.4,
            'feature_2': 0.3,
            'feature_3': 0.2,
            'feature_4': 0.1
        }
        
        fig = evaluator.plot_feature_importance(feature_importance)
        assert fig is not None
        
    def test_plot_feature_importance_empty(self):
        """Test feature importance with empty data."""
        evaluator = ModelEvaluator()
        
        fig = evaluator.plot_feature_importance({})
        assert fig is None
        
    def test_plot_model_comparison(self, sample_evaluation_results):
        """Test model comparison plotting."""
        evaluator = ModelEvaluator()
        
        fig = evaluator.plot_model_comparison(sample_evaluation_results)
        assert fig is not None
        
    def test_plot_model_comparison_empty(self):
        """Test model comparison with empty results."""
        evaluator = ModelEvaluator()
        
        fig = evaluator.plot_model_comparison({})
        assert fig is None
        
    def test_create_evaluation_report(self, sample_evaluation_results):
        """Test comprehensive evaluation report creation."""
        evaluator = ModelEvaluator()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = evaluator.create_evaluation_report(
                sample_evaluation_results,
                output_dir=temp_dir
            )
            
            assert Path(report_path).exists()
            
            # Check report content
            with open(report_path, 'r') as f:
                content = f.read()
            
            assert "Model Evaluation Report" in content
            assert "Random Forest" in content
            assert "Logistic Regression" in content
            assert "Test Accuracy" in content
            assert "Recommendations" in content


if __name__ == "__main__":
    pytest.main([__file__])
