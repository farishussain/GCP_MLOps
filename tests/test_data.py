"""
Tests for data processing modules.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.validator import DataValidator, ValidationResult


class TestDataLoader:
    """Tests for DataLoader class."""
    
    def test_load_iris_dataset(self):
        """Test loading Iris dataset."""
        loader = DataLoader()
        df, metadata = loader.load_iris_dataset()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 150
        assert len(df.columns) == 6  # 4 features + target + target_name
        assert 'target' in df.columns
        assert 'target_name' in df.columns
        
        assert isinstance(metadata, dict)
        assert metadata['n_samples'] == 150
        assert metadata['n_features'] == 4
        assert metadata['n_classes'] == 3
    
    def test_load_from_csv(self):
        """Test loading from CSV file."""
        # Create temporary CSV file
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [5, 6, 7, 8],
            'target': [0, 0, 1, 1]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            loader = DataLoader()
            loaded_df = loader.load_from_csv(temp_file)
            
            assert isinstance(loaded_df, pd.DataFrame)
            assert len(loaded_df) == 4
            assert list(loaded_df.columns) == ['feature1', 'feature2', 'target']
            
        finally:
            Path(temp_file).unlink()
    
    def test_load_from_numpy(self):
        """Test loading from NumPy file."""
        # Create temporary NumPy file
        test_arrays = {
            'X': np.array([[1, 2], [3, 4], [5, 6]]),
            'y': np.array([0, 1, 0])
        }
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            temp_file = f.name
        
        # Save arrays to the temp file
        np.savez(temp_file, X=test_arrays['X'], y=test_arrays['y'])
        
        try:
            loader = DataLoader()
            loaded_arrays = loader.load_from_numpy(temp_file)
            
            assert isinstance(loaded_arrays, dict)
            assert 'X' in loaded_arrays
            assert 'y' in loaded_arrays
            np.testing.assert_array_equal(loaded_arrays['X'], test_arrays['X'])
            np.testing.assert_array_equal(loaded_arrays['y'], test_arrays['y'])
            
        finally:
            Path(temp_file).unlink()


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'feature3': [0.1, 0.2, 0.3, 0.4, 0.5],
            'target': [0, 0, 1, 1, 1]
        })
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = DataPreprocessor(random_state=42)
        
        assert preprocessor.random_state == 42
        assert not preprocessor._is_fitted
    
    def test_create_train_test_split(self, sample_data):
        """Test train/test split creation."""
        preprocessor = DataPreprocessor(random_state=42)
        
        X_train, X_test, y_train, y_test = preprocessor.create_train_test_split(
            sample_data, target_column='target', test_size=0.4
        )
        
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        
        assert len(X_train) == 3  # 60% of 5
        assert len(X_test) == 2   # 40% of 5
        assert len(y_train) == 3
        assert len(y_test) == 2
        
        # Check that target column was removed from features
        assert 'target' not in X_train.columns
        assert 'target' not in X_test.columns
    
    def test_fit_transform_features(self, sample_data):
        """Test feature fitting and transformation."""
        preprocessor = DataPreprocessor(random_state=42)
        
        X_train, _, _, _ = preprocessor.create_train_test_split(
            sample_data, target_column='target'
        )
        
        X_train_processed = preprocessor.fit_transform_features(X_train)
        
        assert isinstance(X_train_processed, pd.DataFrame)
        assert preprocessor._is_fitted
        
        # Check that scaling was applied (mean should be close to 0)
        numeric_cols = X_train_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert abs(X_train_processed[col].mean()) < 1e-10  # Close to 0
    
    def test_transform_features(self, sample_data):
        """Test feature transformation after fitting."""
        preprocessor = DataPreprocessor(random_state=42)
        
        X_train, X_test, _, _ = preprocessor.create_train_test_split(
            sample_data, target_column='target'
        )
        
        # Fit on training data
        preprocessor.fit_transform_features(X_train)
        
        # Transform test data
        X_test_processed = preprocessor.transform_features(X_test)
        
        assert isinstance(X_test_processed, pd.DataFrame)
        assert len(X_test_processed) == len(X_test)
        assert list(X_test_processed.columns) == list(X_test.columns)
    
    def test_encode_target(self, sample_data):
        """Test target encoding."""
        preprocessor = DataPreprocessor()
        
        # Test numeric target
        numeric_target = sample_data['target']
        encoded_numeric = preprocessor.encode_target(numeric_target)
        assert isinstance(encoded_numeric, np.ndarray)
        np.testing.assert_array_equal(encoded_numeric, numeric_target.values)
        
        # Test categorical target
        categorical_target = pd.Series(['cat', 'dog', 'cat', 'bird', 'dog'])
        encoded_categorical = preprocessor.encode_target(categorical_target)
        assert isinstance(encoded_categorical, np.ndarray)
        assert len(encoded_categorical) == len(categorical_target)
        assert len(np.unique(encoded_categorical)) == 3  # 3 unique classes
    
    def test_get_feature_statistics(self, sample_data):
        """Test feature statistics generation."""
        preprocessor = DataPreprocessor()
        
        X_train, _, _, _ = preprocessor.create_train_test_split(
            sample_data, target_column='target'
        )
        
        stats = preprocessor.get_feature_statistics(X_train)
        
        assert isinstance(stats, dict)
        assert 'n_samples' in stats
        assert 'n_features' in stats
        assert 'feature_names' in stats
        assert 'numeric_features' in stats
        assert 'categorical_features' in stats
        assert 'missing_values' in stats
        assert 'data_types' in stats


class TestDataValidator:
    """Tests for DataValidator class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'feature3': [0.1, 0.2, 0.3, 0.4, 0.5],
            'target': [0, 0, 1, 1, 1]
        })
    
    @pytest.fixture
    def problematic_data(self):
        """Create data with quality issues for testing."""
        return pd.DataFrame({
            'feature1': [1.0, 1.0, 1.0, 1.0, 1.0],  # Constant
            'feature2': [10.0, 20.0, None, 40.0, 50.0],  # Missing values
            'feature3': [0.1, 0.2, 0.3, 0.4, 0.5],
            'feature3': [0.6, 0.7, 0.8, 0.9, 1.0],  # Duplicate column name
            'target': [0, 0, 1, 1, 1]
        })
    
    def test_initialization(self):
        """Test validator initialization."""
        validator = DataValidator(strict_mode=True)
        
        assert validator.strict_mode is True
        assert len(validator.validation_history) == 0
    
    def test_validate_dataframe_clean_data(self, sample_data):
        """Test validation with clean data."""
        validator = DataValidator(strict_mode=False)
        
        results = validator.validate_dataframe(sample_data)
        
        assert isinstance(results, list)
        assert all(isinstance(r, ValidationResult) for r in results)
        
        # Most validations should pass for clean data
        passed_count = sum(1 for r in results if r.passed)
        assert passed_count > len(results) * 0.8  # At least 80% should pass
    
    def test_validate_dataframe_with_schema(self, sample_data):
        """Test validation with schema definition."""
        validator = DataValidator(strict_mode=False)
        
        schema = {
            'columns': ['feature1', 'feature2', 'feature3', 'target'],
            'required': ['feature1', 'target'],
            'types': {
                'feature1': 'float64',
                'target': 'int64'
            }
        }
        
        results = validator.validate_dataframe(sample_data, schema=schema)
        
        assert isinstance(results, list)
        # Should find required columns
        required_check = next((r for r in results if 'required columns' in r.message.lower()), None)
        assert required_check is not None
        assert required_check.passed
    
    def test_create_data_profile(self, sample_data):
        """Test data profile creation."""
        validator = DataValidator()
        
        profile = validator.create_data_profile(sample_data)
        
        assert isinstance(profile, dict)
        assert 'basic_info' in profile
        assert 'data_types' in profile
        assert 'missing_values' in profile
        assert 'unique_counts' in profile
        
        basic_info = profile['basic_info']
        assert basic_info['n_rows'] == 5
        assert basic_info['n_columns'] == 4
        assert 'memory_usage_mb' in basic_info
        assert 'column_names' in basic_info
    
    def test_compare_profiles(self, sample_data):
        """Test profile comparison for drift detection."""
        validator = DataValidator()
        
        # Create two similar profiles
        profile1 = validator.create_data_profile(sample_data)
        
        # Modify data slightly
        modified_data = sample_data.copy()
        modified_data['new_feature'] = [1, 2, 3, 4, 5]
        profile2 = validator.create_data_profile(modified_data)
        
        comparison = validator.compare_profiles(profile1, profile2)
        
        assert isinstance(comparison, dict)
        assert 'basic_changes' in comparison
        assert 'schema_changes' in comparison
        assert 'statistical_changes' in comparison
        
        # Should detect the new column
        assert comparison['basic_changes']['columns_change'] == 1
        assert 'new_feature' in comparison['schema_changes']['added_columns']
    
    def test_validation_history(self, sample_data):
        """Test validation history tracking."""
        validator = DataValidator()
        
        # Run multiple validations
        validator.validate_dataframe(sample_data)
        validator.validate_dataframe(sample_data)
        
        assert len(validator.validation_history) == 2
        
        summary = validator.get_validation_summary()
        assert isinstance(summary, dict)
        assert summary['total_validations'] == 2
