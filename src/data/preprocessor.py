"""
Data preprocessing utilities for the MLOps pipeline.

This module provides classes for data preprocessing, feature engineering,
and data transformations.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from ..utils import setup_logging

logger = setup_logging()


class DataPreprocessor:
    """
    Data preprocessing pipeline for ML workflows.
    
    Handles:
    - Train/validation/test splits
    - Feature scaling and normalization
    - Missing value imputation
    - Categorical encoding
    - Feature selection
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize DataPreprocessor.
        
        Args:
            random_state (int): Random state for reproducibility.
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='mean')
        self._is_fitted = False
        
        logger.info(f"Initialized DataPreprocessor with random_state={random_state}")
    
    def create_train_test_split(
        self, 
        df: pd.DataFrame, 
        target_column: str,
        test_size: float = 0.2,
        stratify: bool = True,
        drop_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Create train/test split from DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            target_column (str): Name of target column.
            test_size (float): Proportion of test set.
            stratify (bool): Whether to stratify split by target.
            drop_columns (List[str]): Columns to drop from features.
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test.
        """
        logger.info(f"Creating train/test split with test_size={test_size}")
        
        # Prepare features and target
        X = df.copy()
        y = X.pop(target_column)
        
        # Drop specified columns
        if drop_columns:
            X = X.drop(columns=drop_columns, errors='ignore')
            logger.info(f"Dropped columns: {drop_columns}")
        
        # Create split
        stratify_param = y if stratify and self._is_classification_target(y) else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        logger.info(f"Created split: Train={len(X_train)}, Test={len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def fit_transform_features(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """
        Fit preprocessor on training data and transform features.
        
        Args:
            X_train (pd.DataFrame): Training features.
            
        Returns:
            pd.DataFrame: Transformed training features.
        """
        logger.info("Fitting preprocessor on training data...")
        
        X_train_processed = X_train.copy()
        
        # Handle missing values
        if X_train_processed.isnull().sum().sum() > 0:
            logger.info("Imputing missing values...")
            numeric_columns = X_train_processed.select_dtypes(include=[np.number]).columns
            X_train_processed[numeric_columns] = self.imputer.fit_transform(
                X_train_processed[numeric_columns]
            )
        
        # Scale numeric features
        numeric_columns = X_train_processed.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            logger.info(f"Scaling {len(numeric_columns)} numeric features...")
            X_train_processed[numeric_columns] = self.scaler.fit_transform(
                X_train_processed[numeric_columns]
            )
        
        self._is_fitted = True
        logger.info("Preprocessor fitted successfully")
        
        return X_train_processed
    
    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted preprocessor.
        
        Args:
            X (pd.DataFrame): Features to transform.
            
        Returns:
            pd.DataFrame: Transformed features.
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming")
        
        logger.info("Transforming features...")
        
        X_processed = X.copy()
        
        # Handle missing values
        numeric_columns = X_processed.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0 and X_processed[numeric_columns].isnull().sum().sum() > 0:
            X_processed[numeric_columns] = self.imputer.transform(
                X_processed[numeric_columns]
            )
        
        # Scale numeric features
        if len(numeric_columns) > 0:
            X_processed[numeric_columns] = self.scaler.transform(
                X_processed[numeric_columns]
            )
        
        return X_processed
    
    def encode_target(self, y: pd.Series) -> np.ndarray:
        """
        Encode categorical target variable.
        
        Args:
            y (pd.Series): Target variable.
            
        Returns:
            np.ndarray: Encoded target.
        """
        if y.dtype == 'object' or isinstance(y.iloc[0], str):
            logger.info("Encoding categorical target variable...")
            encoded = self.label_encoder.fit_transform(y)
            return np.array(encoded)
        else:
            logger.info("Target variable is already numeric")
            return np.array(y.values)
    
    def get_feature_statistics(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive feature statistics.
        
        Args:
            X (pd.DataFrame): Features DataFrame.
            
        Returns:
            Dict[str, Any]: Feature statistics.
        """
        stats = {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'feature_names': X.columns.tolist(),
            'numeric_features': X.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_features': X.select_dtypes(include=[object]).columns.tolist(),
            'missing_values': X.isnull().sum().to_dict(),
            'data_types': X.dtypes.astype(str).to_dict()
        }
        
        # Add descriptive statistics for numeric features
        numeric_stats = X.select_dtypes(include=[np.number]).describe()
        if not numeric_stats.empty:
            stats['numeric_stats'] = numeric_stats.to_dict()
        
        return stats
    
    def detect_outliers(self, X: pd.DataFrame, method: str = 'iqr') -> Dict[str, List[int]]:
        """
        Detect outliers in numeric features.
        
        Args:
            X (pd.DataFrame): Features DataFrame.
            method (str): Outlier detection method ('iqr' or 'zscore').
            
        Returns:
            Dict[str, List[int]]: Outlier indices for each feature.
        """
        outliers = {}
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if method == 'iqr':
                Q1 = X[column].quantile(0.25)
                Q3 = X[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = (X[column] < lower_bound) | (X[column] > upper_bound)
            elif method == 'zscore':
                z_scores = np.abs((X[column] - X[column].mean()) / X[column].std())
                outlier_mask = z_scores > 3
            else:
                raise ValueError(f"Unsupported outlier detection method: {method}")
            
            outliers[column] = X[outlier_mask].index.tolist()
        
        return outliers
    
    def create_feature_correlation_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create correlation matrix for numeric features.
        
        Args:
            X (pd.DataFrame): Features DataFrame.
            
        Returns:
            pd.DataFrame: Correlation matrix.
        """
        numeric_features = X.select_dtypes(include=[np.number])
        return numeric_features.corr()
    
    def _is_classification_target(self, y: pd.Series) -> bool:
        """Check if target variable is for classification."""
        unique_values = y.nunique()
        return unique_values < len(y) * 0.1 and unique_values < 20
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get summary of preprocessing steps applied.
        
        Returns:
            Dict[str, Any]: Preprocessing summary.
        """
        summary = {
            'fitted': self._is_fitted,
            'random_state': self.random_state,
            'scaler_type': type(self.scaler).__name__,
            'imputer_strategy': getattr(self.imputer, 'strategy', None),
            'has_label_encoder': hasattr(self, 'label_encoder')
        }
        
        if self._is_fitted:
            scaler_mean = getattr(self.scaler, 'mean_', None)
            scaler_scale = getattr(self.scaler, 'scale_', None)
            imputer_stats = getattr(self.imputer, 'statistics_', None)
            
            summary.update({
                'scaler_mean': scaler_mean.tolist() if scaler_mean is not None else None,
                'scaler_scale': scaler_scale.tolist() if scaler_scale is not None else None,
                'imputer_statistics': imputer_stats.tolist() if imputer_stats is not None else None
            })
        
        return summary
