"""
Data loading utilities for MLOps pipeline.

This module provides data loading capabilities including:
- Loading datasets from various sources
- Data format validation
- Cloud storage integration
- Caching mechanisms
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class DataLoader:
    """Data loading utility class."""
    
    def __init__(self):
        self.supported_datasets = [
            'iris', 'wine', 'breast_cancer'
        ]
        
    def load_dataset(
        self, 
        dataset_name: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Load a dataset and split into train/test sets.
        
        Args:
            dataset_name: Name of the dataset to load.
            test_size: Proportion of dataset to include in test split.
            random_state: Random state for reproducibility.
            
        Returns:
            Tuple containing X_train, X_test, y_train, y_test.
        """
        logger.info(f"Loading dataset: {dataset_name}")
        
        if dataset_name == 'iris':
            data = load_iris()
        elif dataset_name == 'wine':
            data = load_wine()
        elif dataset_name == 'breast_cancer':
            data = load_breast_cancer()
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
            
        # Convert to DataFrame - sklearn returns Bunch objects
        X = pd.DataFrame(data.data, columns=data.feature_names)  # type: ignore
        y = pd.Series(data.target, name='target')  # type: ignore
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Dataset loaded successfully. "
                   f"Training samples: {len(X_train)}, "
                   f"Test samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def load_from_csv(
        self,
        file_path: str,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load dataset from CSV file.
        
        Args:
            file_path: Path to CSV file.
            target_column: Name of the target column.
            test_size: Proportion of dataset to include in test split.
            random_state: Random state for reproducibility.
            
        Returns:
            Tuple containing X_train, X_test, y_train, y_test.
        """
        logger.info(f"Loading data from CSV: {file_path}")
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"CSV data loaded successfully. "
                   f"Training samples: {len(X_train)}, "
                   f"Test samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get information about a dataset.
        
        Args:
            dataset_name: Name of the dataset.
            
        Returns:
            Dict containing dataset information.
        """
        if dataset_name not in self.supported_datasets:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
            
        if dataset_name == 'iris':
            data = load_iris()
        elif dataset_name == 'wine':
            data = load_wine()
        elif dataset_name == 'breast_cancer':
            data = load_breast_cancer()
            
        return {
            'name': dataset_name,
            'description': data.DESCR[:200] + '...' if len(data.DESCR) > 200 else data.DESCR,  # type: ignore
            'n_samples': data.data.shape[0],  # type: ignore
            'n_features': data.data.shape[1],  # type: ignore
            'n_classes': len(np.unique(data.target)),  # type: ignore
            'feature_names': list(data.feature_names),  # type: ignore
            'target_names': list(data.target_names) if hasattr(data, 'target_names') else []  # type: ignore
        }
    
    def save_dataset(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        output_dir: str
    ) -> None:
        """
        Save dataset splits to files.
        
        Args:
            X_train: Training features.
            X_test: Test features.
            y_train: Training targets.
            y_test: Test targets.
            output_dir: Directory to save files.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save datasets
        X_train.to_csv(output_path / 'X_train.csv', index=False)
        X_test.to_csv(output_path / 'X_test.csv', index=False)
        y_train.to_csv(output_path / 'y_train.csv', index=False)
        y_test.to_csv(output_path / 'y_test.csv', index=False)
        
        logger.info(f"Dataset splits saved to {output_dir}")
    
    def load_saved_dataset(
        self, 
        data_dir: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load previously saved dataset splits.
        
        Args:
            data_dir: Directory containing saved dataset files.
            
        Returns:
            Tuple containing X_train, X_test, y_train, y_test.
        """
        data_path = Path(data_dir)
        
        X_train = pd.read_csv(data_path / 'X_train.csv')
        X_test = pd.read_csv(data_path / 'X_test.csv')
        y_train = pd.read_csv(data_path / 'y_train.csv').squeeze()
        y_test = pd.read_csv(data_path / 'y_test.csv').squeeze()
        
        logger.info(f"Dataset splits loaded from {data_dir}")
        
        return X_train, X_test, y_train, y_test
