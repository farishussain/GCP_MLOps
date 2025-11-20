"""
Data preprocessing utilities for MLOps pipeline.

This module provides data preprocessing capabilities including:
- Feature scaling and normalization
- Categorical encoding
- Feature engineering
- Data transformation pipelines
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

from ..utils import save_json, load_json

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Data preprocessing utility class."""
    
    def __init__(self):
        self.preprocessors = {}
        self.feature_names = None
        self.is_fitted = False
        
    def create_preprocessing_pipeline(
        self,
        numeric_features: List[str],
        categorical_features: List[str],
        scale_method: str = 'standard',
        handle_missing: bool = True,
        encode_categorical: bool = True
    ) -> Pipeline:
        """
        Create a preprocessing pipeline.
        
        Args:
            numeric_features: List of numeric feature names.
            categorical_features: List of categorical feature names.
            scale_method: Scaling method ('standard', 'minmax', 'none').
            handle_missing: Whether to handle missing values.
            encode_categorical: Whether to encode categorical variables.
            
        Returns:
            Sklearn preprocessing pipeline.
        """
        logger.info("Creating preprocessing pipeline...")
        
        # Numeric pipeline
        numeric_steps = []
        
        if handle_missing:
            numeric_steps.append(('imputer', SimpleImputer(strategy='mean')))
            
        if scale_method == 'standard':
            numeric_steps.append(('scaler', StandardScaler()))
        elif scale_method == 'minmax':
            numeric_steps.append(('scaler', MinMaxScaler()))
        
        numeric_pipeline = Pipeline(numeric_steps) if numeric_steps else 'passthrough'
        
        # Categorical pipeline
        categorical_steps = []
        
        if handle_missing:
            categorical_steps.append(('imputer', SimpleImputer(strategy='constant', fill_value='missing')))
            
        if encode_categorical:
            categorical_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
        
        categorical_pipeline = Pipeline(categorical_steps) if categorical_steps else 'passthrough'
        
        # Combine pipelines
        transformers = []
        if numeric_features:
            transformers.append(('num', numeric_pipeline, numeric_features))
        if categorical_features:
            transformers.append(('cat', categorical_pipeline, categorical_features))
            
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor)
        ])
        
        logger.info(f"Pipeline created with {len(numeric_features)} numeric and "
                   f"{len(categorical_features)} categorical features")
        
        return pipeline
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Pipeline]:
        """
        Fit preprocessing pipeline and transform data.
        
        Args:
            X: Input features DataFrame.
            y: Target variable (optional).
            **kwargs: Additional arguments for pipeline creation.
            
        Returns:
            Tuple of (transformed_data, fitted_pipeline).
        """
        logger.info("Fitting and transforming data...")
        
        # Identify feature types
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Identified {len(numeric_features)} numeric and "
                   f"{len(categorical_features)} categorical features")
        
        # Create pipeline
        pipeline = self.create_preprocessing_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            **kwargs
        )
        
        # Fit and transform
        X_transformed = pipeline.fit_transform(X)
        
        # Store pipeline and metadata
        self.preprocessors['main'] = pipeline
        self.feature_names = self._get_feature_names(pipeline, X)
        self.is_fitted = True
        
        logger.info(f"Data transformed from {X.shape} to {X_transformed.shape}")
        
        return X_transformed, pipeline
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted pipeline.
        
        Args:
            X: Input features DataFrame.
            
        Returns:
            Transformed data array.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        pipeline = self.preprocessors['main']
        X_transformed = pipeline.transform(X)
        
        logger.info(f"Data transformed from {X.shape} to {X_transformed.shape}")
        
        return X_transformed
    
    def _get_feature_names(self, pipeline: Pipeline, X: pd.DataFrame) -> List[str]:
        """
        Get feature names after transformation.
        
        Args:
            pipeline: Fitted preprocessing pipeline.
            X: Original DataFrame.
            
        Returns:
            List of feature names.
        """
        try:
            # Try to get feature names from the pipeline
            preprocessor = pipeline.named_steps['preprocessor']
            
            feature_names = []
            
            # Get names for each transformer
            for name, transformer, features in preprocessor.transformers_:
                if hasattr(transformer, 'get_feature_names_out'):
                    if name == 'num':
                        names = [f"num_{feat}" for feat in features]
                    elif name == 'cat':
                        names = transformer.get_feature_names_out(features).tolist()
                    else:
                        names = transformer.get_feature_names_out(features).tolist()
                    feature_names.extend(names)
                else:
                    # Fallback to original feature names
                    feature_names.extend(features)
            
            return feature_names
            
        except Exception as e:
            logger.warning(f"Could not get feature names: {e}")
            # Fallback to generic names
            n_features = pipeline.transform(X.iloc[:1]).shape[1]
            return [f"feature_{i}" for i in range(n_features)]
    
    def handle_missing_values(
        self,
        X: pd.DataFrame,
        strategy: str = 'auto'
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            X: Input DataFrame.
            strategy: Strategy for handling missing values ('auto', 'drop', 'impute').
            
        Returns:
            DataFrame with missing values handled.
        """
        logger.info("Handling missing values...")
        
        missing_counts = X.isnull().sum()
        missing_percentage = (missing_counts / len(X)) * 100
        
        if strategy == 'auto':
            # Auto-select strategy based on missing percentage
            X_processed = X.copy()
            
            for col in X.columns:
                col_missing_pct = missing_percentage[col]
                
                if col_missing_pct > 50:
                    # Drop columns with >50% missing values
                    logger.info(f"Dropping column {col} ({col_missing_pct:.1f}% missing)")
                    X_processed = X_processed.drop(columns=[col])
                elif col_missing_pct > 0:
                    # Impute missing values
                    if X_processed[col].dtype in ['object', 'category']:
                        # Most frequent for categorical
                        X_processed[col] = X_processed[col].fillna(X_processed[col].mode().iloc[0])
                    else:
                        # Mean for numeric
                        X_processed[col] = X_processed[col].fillna(X_processed[col].mean())
                        
        elif strategy == 'drop':
            # Drop rows with any missing values
            X_processed = X.dropna()
            
        elif strategy == 'impute':
            # Impute all missing values
            X_processed = X.copy()
            for col in X.columns:
                if X_processed[col].isnull().any():
                    if X_processed[col].dtype in ['object', 'category']:
                        X_processed[col] = X_processed[col].fillna('missing')
                    else:
                        X_processed[col] = X_processed[col].fillna(X_processed[col].median())
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        logger.info(f"Missing values handled. Shape: {X.shape} -> {X_processed.shape}")
        
        return X_processed
    
    def encode_categorical_features(
        self,
        X: pd.DataFrame,
        method: str = 'onehot',
        max_categories: int = 10
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Encode categorical features.
        
        Args:
            X: Input DataFrame.
            method: Encoding method ('onehot', 'label', 'target').
            max_categories: Maximum number of categories for one-hot encoding.
            
        Returns:
            Tuple of (encoded_dataframe, encoders_dict).
        """
        logger.info(f"Encoding categorical features using {method} method...")
        
        X_encoded = X.copy()
        encoders = {}
        
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            unique_values = X[col].nunique()
            
            if method == 'onehot' and unique_values <= max_categories:
                # One-hot encoding
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_values = encoder.fit_transform(X[[col]])
                
                # Create column names
                feature_names = [f"{col}_{val}" for val in encoder.categories_[0]]  # type: ignore
                
                # Create DataFrame with encoded features
                encoded_df = pd.DataFrame(encoded_values, columns=feature_names, index=X.index)
                
                # Replace original column
                X_encoded = X_encoded.drop(columns=[col])
                X_encoded = pd.concat([X_encoded, encoded_df], axis=1)
                
                encoders[col] = encoder
                
            elif method == 'label' or (method == 'onehot' and unique_values > max_categories):
                # Label encoding
                encoder = LabelEncoder()
                X_encoded[col] = encoder.fit_transform(X[col].astype(str))
                encoders[col] = encoder
                
        logger.info(f"Categorical encoding complete. Shape: {X.shape} -> {X_encoded.shape}")
        
        return X_encoded, encoders
    
    def create_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features through feature engineering.
        
        Args:
            X: Input DataFrame.
            
        Returns:
            DataFrame with additional features.
        """
        logger.info("Creating additional features...")
        
        X_features = X.copy()
        
        # Numeric features for feature engineering
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) >= 2:
            # Create interaction features for first few numeric columns
            for i, col1 in enumerate(numeric_columns[:3]):
                for col2 in numeric_columns[i+1:4]:
                    # Interaction terms
                    X_features[f"{col1}_x_{col2}"] = X[col1] * X[col2]
                    
                    # Ratio features (avoid division by zero)
                    X_features[f"{col1}_div_{col2}"] = X[col1] / (X[col2] + 1e-8)
        
        # Statistical features
        if len(numeric_columns) > 0:
            # Row-wise statistics
            X_features['row_sum'] = X[numeric_columns].sum(axis=1)
            X_features['row_mean'] = X[numeric_columns].mean(axis=1)
            X_features['row_std'] = X[numeric_columns].std(axis=1)
            X_features['row_max'] = X[numeric_columns].max(axis=1)
            X_features['row_min'] = X[numeric_columns].min(axis=1)
        
        logger.info(f"Feature engineering complete. Shape: {X.shape} -> {X_features.shape}")
        
        return X_features
    
    def save_preprocessor(self, output_path: str) -> None:
        """
        Save fitted preprocessor to disk.
        
        Args:
            output_path: Path to save the preprocessor.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the main pipeline
        joblib.dump(self.preprocessors['main'], output_path_obj)
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        metadata_path = output_path_obj.parent / f"{output_path_obj.stem}_metadata.json"
        save_json(metadata, str(metadata_path))
        
        logger.info(f"Preprocessor saved to {output_path_obj}")
    
    def load_preprocessor(self, model_path: str) -> None:
        """
        Load saved preprocessor from disk.
        
        Args:
            model_path: Path to the saved preprocessor.
        """
        model_path_obj = Path(model_path)
        
        # Load the main pipeline
        self.preprocessors['main'] = joblib.load(model_path_obj)
        
        # Load metadata
        metadata_path = model_path_obj.parent / f"{model_path_obj.stem}_metadata.json"
        if metadata_path.exists():
            metadata = load_json(str(metadata_path))
            self.feature_names = metadata.get('feature_names')
            self.is_fitted = metadata.get('is_fitted', True)
        else:
            self.is_fitted = True
            
        logger.info(f"Preprocessor loaded from {model_path}")
    
    def get_feature_importance_from_preprocessing(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance information from preprocessing steps.
        
        Returns:
            Dictionary of feature importance or None.
        """
        if not self.is_fitted or not self.feature_names:
            return None
        
        # For now, return uniform importance
        # This could be enhanced to analyze variance, correlation, etc.
        n_features = len(self.feature_names)
        importance = {name: 1.0 / n_features for name in self.feature_names}
        
        return importance
